import pylab
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class GraddescentMinibatch(object):

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, normalizefilters=True, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        self.normalizefilters = normalizefilters
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        def inplaceclip(x):
            x[:,:] *= x>0.0
            return x

        def inplacemask(x, mask):
            x[:,:] *= mask
            return x

        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            if self.normalizefilters:
                self.model.layer.normalizefilters()
            #self.model.layer.whf.set_value(inplaceclip(self.model.layer.whf.get_value(borrow=True)), borrow=True)
            #self.model.layer.whf.set_value(inplacemask(self.model.layer.whf.get_value(borrow=True), self.model.layer.topomask), borrow=True)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)


class GraddescentMinibatch_unloaded(object):
    def __init__(self, model, features_numpy, batchsize, learningrate, momentum=0.9, loadsize=None, rng=None, verbose=True):
        self.model          = model
        self.features_numpy = features_numpy
        self.learningrate   = learningrate
        self.verbose        = verbose
        self.batchsize      = batchsize
        if loadsize == None:
            loadsize = batchsize * 100
        self.loadsize      = loadsize 
        self.numloads      = self.features_numpy.shape[0] / self.loadsize
        self.features      = theano.shared(self.features_numpy[:self.loadsize])
        self.numbatches    = self.loadsize / batchsize
        self.momentum      = momentum 
        self.momentum_batchcounter = 0
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates_nomomentum = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            # Non-cliphid version:
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
            self.updates[_param] = _param + self.incs[_param]
            self.updates_nomomentum[_param] = _param - self.learningrate * _grad 

        # Non-cliphid version:
        self._updateincs = theano.function([self.index], self.model._cost, updates = self.inc_updates, 
          givens = {self.model.inputs:self.features[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)   

        self._trainmodel_nomomentum = theano.function([self.n, self.index], self.noop, updates = self.updates_nomomentum, 
          givens = {self.model.inputs:self.features[self.index*self.batchsize:(self.index+1)*self.batchsize]})

        self.momentum_batchcounter = 0

    def step(self):
        cost = 0.0
        stepcount = 0.0
        self.epochcount += 1
        for load_index in range(self.numloads):
            self.features.set_value(self.features_numpy[load_index * self.loadsize:(load_index + 1) * self.loadsize])
            for batch_index in self.rng.permutation(self.numbatches-1):
            #for batch_index in range(self.numbatches):
                stepcount += 1.0
                self.momentum_batchcounter += 1
                if self.momentum_batchcounter < 10:
                    cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
                    self._trainmodel_nomomentum(0, batch_index)
                else: 
                    self.momentum_batchcounter = 10
                    cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
                    self._trainmodel(0)

            if self.verbose:
                print '> epoch %d, load %d, cost: %r' % (self.epochcount, load_index + 1, cost)
            if numpy.isnan(cost):
                raise ValueError, 'Cost function returned nan!'
            elif numpy.isinf(cost):
                raise ValueError, 'Cost function returned infinity!'


