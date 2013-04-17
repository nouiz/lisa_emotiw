import numpy
import time

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import safe_clone, print_time, print_mem, const


class SGD(object):
    def __init__(self,
                 model,
                 state,
                 data):
        """
        Parameters:
            :param model:
                Class describing the model used.  It should provide the
                 computational graph to evaluate the model
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
        """

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        n_params = len(model.params)
        bs = state['bs']
        profile = state['profile']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])

        self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]

        self.bs = bs
        self.state = state
        self.profile = profile
        self.data = data
        self.data.set_iterator(order = 'sequence',
                               rng = self.rng,
                               batchsize = self.state['bs'])
        self.data_iter = data.__iter__()
        self.step_timer = time.time()

        ############################################################
        # Step 1. Compile function for computing eucledian gradients
        ############################################################
        print 'Constructing grad function'
        gs = TT.grad(self.model.train_cost, model.params)
        update = [(g, lg) for g, lg in zip(self.gs, gs)]
        print 'Compiling grad function'
        st = time.time()
        self.grad_fn = theano.function(
            self.model.inputs, [], updates=update, name='loc_fn_grad', profile=profile)
        print 'took', time.time() - st

        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))
        ###########################################################
        # Step 3. Compile function for evaluating cost and updating
        # parameters
        ###########################################################
        print 'constructing evaluation function'
        lr = TT.scalar('lr')
        self.lr = numpy.float32(state['lr'])
        old_cost = model.train_cost
        self.compute_old_cost = theano.function(
            self.model.inputs, old_cost, name='loc_old_cost', profile=profile)
        new_params = [p - lr * r for p, r in zip(model.params, self.gs)]
        new_cost = safe_clone(model.train_cost,
                              model.params, new_params)
        new_err = safe_clone(model.error,
                             model.params, new_params)
        self.compute_new_cost = theano.function(
            [lr]+self.model.inputs, [new_cost, new_err], name='loc_new_cost',
            profile=profile)

        self.update_params = theano.function(
            [lr], [], updates=zip(model.params, new_params),
            name='update_params')
        old_cost = TT.scalar('old_cost')
        new_cost = TT.scalar('new_cost')
        dist = -lr * sum([TT.sum(g * r) for g, r in zip(self.gs, self.gs)])
        rho = (new_cost - old_cost) / dist
        self.compute_rho = theano.function(
            [old_cost, new_cost, lr], [rho, norm_grads], name='compute_rho', profile=profile)
        self.old_cost = 1e20
        self.step = 0
        self.return_names = ['cost',
                             'error',
                             'time_grads',
                             'time_eval',
                             'norm_grad',
                             'rho',
                             'lr']


    def __call__(self):

        batch = self.data_iter.get_batch()
        g_st = time.time()
        self.grad_fn(*batch)
        g_ed = time.time()
        if self.state['lr_adapt'] == 1:
            if self.step > self.state['lr_adapt_start']:
                self.lr = self.state['lr0'] /\
                    (1. + float(self.step - self.state['lr_adapt_start'])/self.state['lr_beta'])
                self.state['lr'] = float(self.lr)
        e_st = time.time()
        old_cost = self.compute_old_cost(*batch)
        new_cost, error = self.compute_new_cost(self.lr, *batch)
        rho, norm_grad = self.compute_rho(old_cost, new_cost, self.lr)

        if new_cost > old_cost:
            print ('Error increasing !? ')
            self.lr = self.lr / 2.

        while (numpy.isnan(new_cost) or
               numpy.isinf(new_cost)):
            raise Exception('Got Inf/NaN !')
        self.old_cost = new_cost
        self.update_params(self.lr)
        e_ed = time.time()
        msg = ('.. iter %4d cost %.3g (before update %.3g), error %.3g step_size %.3g '
               'rho %.3g '
               'norm grad %.3g '
               'time [grad] %s,'
               '[updates param] %s,'
               'whole time %s'
              )
        print msg % (
            self.step,
            new_cost,
            old_cost,
            error,
            self.lr,
            rho,
            norm_grad,
            print_time(g_ed - g_st),
            print_time(e_ed - e_st),
            print_time(time.time() - self.step_timer) )
        self.step_timer = time.time()
        self.step += 1

        ret = {
            'cost': float(new_cost),
            'error': float(error),
            'time_grads': float(g_ed - g_st),
            'time_eval': float(e_ed - e_st),
            'norm_grad':norm_grad,
            'lr': self.lr,
            'rho': numpy.float32(rho)}
        return ret
