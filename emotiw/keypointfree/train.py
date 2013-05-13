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

