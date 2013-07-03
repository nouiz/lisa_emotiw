"""
This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import numpy
import md5
import pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared
from theano.sandbox import linalg
from theano.ifelse import ifelse
from theano.sandbox.scan import scan

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

import truncated
import cost as costmod
from utils import tools
from utils import rbm_utils
from utils import sharedX, floatX, npy_floatX

def sigm(x): return 1./(1 + numpy.exp(-x))
def softplus(x): return numpy.log(1. + numpy.exp(x))
def softplus_inv(x): return numpy.log(numpy.exp(x) - 1.)

class SpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def validate_flags(self, flags):
        flags.setdefault('truncate_v', False)
        flags.setdefault('scalar_lambd', False)
        flags.setdefault('wh_norm', 'none')
        flags.setdefault('wv_norm', 'none')
        flags.setdefault('ml_lambd', False)
        flags.setdefault('init_mf_rand', False)
        flags.setdefault('center_h', False)
        if len(flags.keys()) != 7:
            raise NotImplementedError('One or more flags are currently not implemented.')

    def __init__(self, 
            numpy_rng = None, theano_rng = None,
            n_h=99, n_s=99, n_v=100, init_from=None,
            sparse_hmask = None,
            neg_sample_steps=1,
            lr_spec=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, truncation_bound={},
            l1 = {}, l2 = {},
            sp_weight={}, sp_targ={},
            batch_size = 13,
            compile=True,
            debug=False,
            seed=1241234,
            my_save_path=None, save_at=None, save_every=None,
            flags = {},
            max_updates = 5e5):
        """
        :param n_h: number of h-hidden units
        :param n_v: number of visible units
        :param iscales: optional dictionary containing initialization scale for each parameter
        :param neg_sample_steps: number of sampling updates to perform in negative phase.
        :param l1: hyper-parameter controlling amount of L1 regularization
        :param l2: hyper-parameter controlling amount of L2 regularization
        :param batch_size: size of positive and negative phase minibatch
        :param compile: compile sampling and learning functions
        :param seed: seed used to initialize numpy and theano RNGs.
        """
        Model.__init__(self)
        Block.__init__(self)
        assert lr_spec is not None
        for k in ['h']: assert k in sp_weight.keys()
        for k in ['h']: assert k in sp_targ.keys()
        self.validate_flags(flags)

        self.jobman_channel = None
        self.jobman_state = {}
        self.register_names_to_del(['jobman_channel'])

        ### make sure all parameters are floatX ###
        for (k,v) in l1.iteritems(): l1[k] = npy_floatX(v)
        for (k,v) in l2.iteritems(): l2[k] = npy_floatX(v)
        for (k,v) in sp_weight.iteritems(): sp_weight[k] = npy_floatX(v)
        for (k,v) in sp_targ.iteritems(): sp_targ[k] = npy_floatX(v)
        for (k,v) in clip_min.iteritems(): clip_min[k] = npy_floatX(v)
        for (k,v) in clip_max.iteritems(): clip_max[k] = npy_floatX(v)

        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # allocate random number generators
        self.rng = numpy.random.RandomState(seed) if numpy_rng is None else numpy_rng
        self.theano_rng = RandomStreams(self.rng.randint(2**30)) if theano_rng is None else theano_rng

        ############### ALLOCATE PARAMETERS #################
        # allocate symbolic variable for input
        self.input = T.matrix('input')
        self.init_parameters()
        self.init_chains()

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr_spec['type'] == 'anneal':
            num = lr_spec['init'] * lr_spec['start'] 
            denum = T.maximum(lr_spec['start'], lr_spec['slope'] * self.iter)
            self.lr = T.maximum(lr_spec['floor'], num/denum) 
        elif lr_spec['type'] == 'linear':
            lr_start = npy_floatX(lr_spec['start'])
            lr_end   = npy_floatX(lr_spec['end'])
            self.lr = lr_start + self.iter * (lr_end - lr_start) / npy_floatX(self.max_updates)
        else:
            raise ValueError('Incorrect value for lr_spec[type]')

        # configure input-space (new pylearn2 feature?)
        self.input_space = VectorSpace(n_v)
        self.output_space = VectorSpace(n_h)

        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size  # force minibatch size

        self.error_record = []
 
        if compile: self.do_theano()

        #### load layer 1 parameters from file ####
        if init_from:
            self.load_params(init_from)

    def init_weight(self, iscale, shape, name, normalize=False, axis=0):
        value = self.rng.normal(size=shape) * iscale
        if normalize:
            value /= numpy.sqrt(numpy.sum(value**2, axis=axis))
        return sharedX(value, name=name)

    def init_parameters(self):
        assert self.sparse_hmask

        # Init (visible, slabs) weight matrix.
        self.Wv = self.init_weight(self.iscales['Wv'], (self.n_v, self.n_s), 'Wv', normalize=True)
        self.norm_wv = T.sqrt(T.sum(self.Wv**2, axis=0))
        self.mu = sharedX(self.iscales['mu'] * numpy.ones(self.n_s), name='mu')

        # Initialize (slab, hidden) pooling matrix
        self.Wh = sharedX(self.sparse_hmask.mask.T * self.iscales.get('Wh', 1.0), name='Wh')

        # allocate shared variables for bias parameters
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.ch = sharedX(0.5 * numpy.ones(self.n_h), name='ch')

        # precision (alpha) parameters on s
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = T.nnet.softplus(self.alpha)

        # diagonal of precision matrix of visible units
        self.lambd = sharedX(self.iscales['lambd'] * numpy.ones(self.n_v), name='lambd')
        self.lambd_prec = T.nnet.softplus(self.lambd)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        # initialize visible unit chains
        scale = numpy.sqrt(1./softplus(self.lambd.get_value()))
        neg_v  = self.rng.normal(loc=0, scale=scale, size=(self.batch_size, self.n_v))
        self.neg_v  = sharedX(neg_v, name='neg_v')
        # initialize s-chain
        scale = numpy.sqrt(1./softplus(self.alpha.get_value()))
        neg_s  = self.rng.normal(loc=0., scale=scale, size=(self.batch_size, self.n_s))
        self.neg_s  = sharedX(neg_s, name='neg_s')
        # initialize binary g-h chains
        pval_h = sigm(self.hbias.get_value())
        neg_h = self.rng.binomial(n=1, p=pval_h, size=(self.batch_size, self.n_h))
        self.neg_h  = sharedX(neg_h, name='neg_h')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.hbias, self.mu, self.alpha, self.lambd]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        self.init_debug()

        # POSITIVE PHASE
        pos_h = self.h_given_v(self.input)
        self.inference_func = theano.function([self.input], [pos_h])

        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps, use_pcd=True)
        self.sample_func = theano.function([], [], updates=neg_updates)

        ##
        # BUILD COST OBJECTS
        ##
        lcost = self.ml_cost(
                        pos_v = self.input,
                        neg_v = neg_updates[self.neg_v])

        regcost = self.get_reg_cost(self.l2, self.l1)

        ##
        # COMPUTE GRADIENTS WRT. COSTS
        ##
        main_cost = [lcost, regcost]

        learning_grads = costmod.compute_gradients(self.lr, self.lr_mults, *main_cost)

        ##
        # BUILD UPDATES DICTIONARY FROM GRADIENTS
        ##
        learning_updates = costmod.get_updates(learning_grads)
        learning_updates.update(neg_updates)
        learning_updates.update({self.iter: self.iter+1})

        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates,
                                         name='train_rbm_func')
        #theano.printing.pydotprint(self.batch_train_func, outfile='batch_train_func.png', scan_graphs=True);

        #######################
        # CONSTRAINT FUNCTION #
        #######################
        constraint_updates = self.get_constraint_updates()
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def get_constraint_updates(self):
        constraint_updates = OrderedDict() 

        constraint_updates[self.Wv] = self.Wv / self.norm_wv

        if self.flags['scalar_lambd']:
            constraint_updates[self.lambd] = T.mean(self.lambd) * T.ones_like(self.lambd)

        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params()]
            param = constraint_updates.get(k, getattr(self, k))
            constraint_updates[param] = T.clip(param, param, v)

        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params()]
            param = constraint_updates.get(k, getattr(self, k))
            constraint_updates[param] = T.clip(constraint_updates.get(param, param), v, param)

        return constraint_updates

    def train_batch(self, dataset, batch_size):

        (x, y) = dataset.get_random_framepair_batch(batch_size)
        if self.flags['truncate_v']:
            x = numpy.clip(x, -self.truncation_bound['v'], self.truncation_bound['v'])

        try:
            self.batch_train_func(x.astype(floatX))
            self.enforce_constraints()
        except:
            import pdb; pdb.set_trace()

        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1

        # save to different path each epoch
        if self.my_save_path and \
           (self.batches_seen in self.save_at or
            self.batches_seen % self.save_every == 0):
            fname = self.my_save_path + '_e%i.pkl' % self.batches_seen
            print 'Saving to %s ...' % fname,
            serial.save(fname, self)
            print 'done'

        return self.batches_seen < self.max_updates

    def energy(self, h_sample, s_sample, v_sample):
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        ch_sample = h_sample - self.ch if self.flags['center_h'] else h_sample

        energy  = 0.
        energy -= T.sum(from_v * self.mu * from_h, axis=1)
        energy -= T.sum(from_v * s_sample * from_h, axis=1)
        energy += 0.5 * T.sum(self.alpha_prec * s_sample**2, axis=1)
        energy += T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        energy -= T.dot(ch_sample, self.hbias)
        return energy, [h_sample, s_sample, v_sample]

    def free_energy(self, v_sample):
        fe  = T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        fe -= 0.5 * T.sum(T.log(2*numpy.pi / self.alpha_prec))
        h_mean = self.h_given_v_input(v_sample)
        fe -= T.sum(T.nnet.softplus(h_mean), axis=1)
        return fe

    def __call__(self, v, output_type='h'):
        print 'Building representation with %s' % output_type
        init_state = OrderedDict()
        h = self.h_given_v(v)
        s = self.s_given_hv(h, v)

        atoms = {
            'h_s' : self.from_h(h),  # h in s-space
            's_h' : T.sqrt(self.to_h(s**2)),
        }

        output_prods = {
            'h' : h,
            's' : s,
            'hs': h * atoms['s_h'],
        }

        toks = output_type.split('+')
        output = output_prods[toks[0]]
        for tok in toks[1:]:
            output = T.horizontal_stack(output, output_prods[tok])

        return output

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def from_v(self, v_sample):
        return T.dot(self.lambd_prec * v_sample, self.Wv)

    def from_h(self, h_sample):
        if self.flags['center_h']:
            h_sample = h_sample - self.ch
        return T.dot(h_sample, self.Wh.T)

    def to_h(self, h_s):
        return T.dot(h_s, self.Wh)

    def h_given_v_input(self, v_sample):
        from_v = self.from_v(v_sample)
        h_mean_s = from_v * self.mu
        h_mean_s += 0.5 * 1./self.alpha_prec * from_v**2
        h_mean = self.to_h(h_mean_s) + self.hbias
        return h_mean
 
    def h_given_v(self, v_sample):
        h_mean = self.h_given_v_input(v_sample)
        return T.nnet.sigmoid(h_mean)
    
    def sample_h_given_v(self, v_sample, rng=None, size=None):
        """
        Generates sample from p(h | v)
        """
        h_mean = self.h_given_v(v_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        h_sample = rng.binomial(size=(size, self.n_h),
                                n=1, p=h_mean, dtype=floatX)
        return h_sample

    def s_given_hv(self, h_sample, v_sample):
        from_h = self.from_h(h_sample)
        from_v = self.from_v(v_sample)
        s_mean = 1./self.alpha_prec * from_v * from_h
        return s_mean

    def sample_s_given_hv(self, h_sample, v_sample, rng=None, size=None):
        """
        Generates sample from p(s | h, v)
        """
        s_mean = self.s_given_hv(h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size

        s_sample = rng.normal(
                size=(size, self.n_s),
                avg = s_mean, 
                std = T.sqrt(1./self.alpha_prec),
                dtype=floatX)

        return s_sample

    def v_given_hs(self, h_sample, s_sample):
        from_h = self.from_h(h_sample)
        v_mean =  T.dot(from_h * (self.mu + s_sample), self.Wv.T)
        return v_mean

    def sample_v_given_hs(self, h_sample, s_sample, rng=None, size=None):
        """
        Generates sample from p(v | h, s)
        """
        v_mean = self.v_given_hs(h_sample, s_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        if self.flags['truncate_v']:
            v_sample = truncated.truncated_normal(
                    size=(size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.lambd_prec),
                    lbound = -self.truncation_bound['v'],
                    ubound = self.truncation_bound['v'],
                    theano_rng = rng,
                    dtype=floatX)
        else:
            v_sample = rng.normal(
                    size=(size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.lambd_prec),
                    dtype=floatX)

        return v_sample

    ##################
    # SAMPLING STUFF #
    ##################
    def neg_sampling(self, h_sample, s_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(h1, s1, v1):
            h2 = self.sample_h_given_v(v1)
            s2 = self.sample_s_given_hv(h2, v1)
            v2 = self.sample_v_given_hs(h2, s2)
            return [h2, s2, v2]

        rvals , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [h_sample, s_sample, v_sample],
                n_steps=n_steps)
        
        return [rval[-1] for rval in rvals]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_h, new_s, new_v] =  self.neg_sampling(
                self.neg_h, self.neg_s, self.neg_v,
                n_steps = n_steps)

        updates = OrderedDict()
        updates[self.neg_h] = new_h
        updates[self.neg_s] = new_s
        updates[self.neg_v] = new_v
        return updates

    def ml_cost(self, pos_v, neg_v):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        pos_cost = T.mean(self.free_energy(pos_v))
        neg_cost = T.mean(self.free_energy(neg_v))
        cost = pos_cost - neg_cost
        return costmod.Cost(cost, self.params(), [pos_v,neg_v])

    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_sparsity_cost(self, pos_g, pos_h):
        raise NotImplementedError()

    def get_reg_cost(self, l2=None, l1=None):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict whose values represent amount of L2 regularization to apply to
        parameter specified by key.
        :param l1: idem for l1.
        """
        cost = T.zeros((), dtype=floatX)
        params = []

        for p in self.params():

            if l1.get(p.name, 0):
                cost += l1[p.name] * T.sum(abs(p))
                params += [p]

            if l2.get(p.name, 0):
                cost += l2[p.name] * T.sum(p**2)
                params += [p]
            
        return costmod.Cost(cost, params)

    def monitor_matrix(self, w, name=None, abs_mean=True):
        if name is None: assert hasattr(w, 'name')
        name = name if name else w.name

        rval = OrderedDict()
        rval[name + '.min'] = w.min(axis=[0,1])
        rval[name + '.max'] = w.max(axis=[0,1])
        if abs_mean:
            rval[name + '.absmean'] = abs(w).mean(axis=[0,1])
        else:
            rval[name + '.mean'] = w.mean(axis=[0,1])
        return rval

    def monitor_vector(self, b, name=None):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name

        rval = OrderedDict()
        rval[name + '.min'] = b.min()
        rval[name + '.max'] = b.max()
        rval[name + '.absmean'] = abs(b).mean()
        return rval

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.lambd_prec, name='lambd_prec'))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s))
        chans.update(self.monitor_matrix(self.neg_v))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        return chans

    def init_debug(self):
        neg_h = self.h_given_v(self.neg_v)
        neg_s = self.s_given_hv(self.neg_h, self.neg_v)
        neg_v = self.v_given_hs(self.neg_h, self.neg_s)
        self.sample_h_func = theano.function([], neg_h)
        self.sample_s_func = theano.function([], neg_s)
        self.sample_v_func = theano.function([], neg_v)

        # Build function to compute energies.
        hh = T.matrix('h')
        ss = T.matrix('s')
        vv = T.matrix('v')
        E, _crap = self.energy(hh,ss,vv)
        self.energy_func = theano.function([hh,ss,vv], E)


import pylab as pl

class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def init_params_from_data(self, model, x):

        if model.flags['ml_lambd']:
            # compute maximum likelihood solution for lambd
            scale = (1./numpy.std(x, axis=0))**2
            model.lambd.set_value(softplus_inv(scale).astype(floatX))
            # reset neg_v markov chain accordingly
            neg_v = model.rng.normal(loc=0, scale=scale, size=(model.batch_size, model.n_v))
            model.neg_v.set_value(neg_v.astype(floatX))

        """
        [pos_g, pos_h, pos_s] = model.inference_func(x)
        if model.flags['center_g']:
            model.cg.set_value(pos_g.mean(axis=0).astype(floatX))
        if model.flags['center_h']:
            model.ch.set_value(pos_h.mean(axis=0).astype(floatX))
  
        [pos_g, pos_h, pos_s] = model.inference_func(x)
        e1 = model.energy_func(pos_g, pos_h, pos_s, x)
        pl.hist(e1, bins=100); pl.show()

        [pos_g, pos_h, pos_s] = model.inference_func(x); e1 = model.energy_func(pos_g, pos_h, pos_s, x); pl.hist(e1, bins=100); pl.show()

        import pdb; pdb.set_trace()
        """


    def setup(self, model, dataset):

        x = dataset.get_batch_design(10000, include_labels=False)
        self.init_params_from_data(model, x)
        super(TrainingAlgorithm, self).setup(model, dataset)
