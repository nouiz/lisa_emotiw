import numpy, pylab
import cPickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from tools import dispims
from minimize import minimize  


class Patchpairencoder(object):
    def __init__(self, numvisD, numvisY, numhid, output_type, 
                 corruption_type='zeromask', corruption_level=0.0, weightcost=0.0, contraction=0.0,
                 numpy_rng=None, theano_rng=None):
        self.numvisD = numvisD
        self.numvisY = numvisY
        self.numhid  = numhid
        self.output_type  = output_type
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.weightcost = weightcost
        self.contraction = contraction
        self.inputs = T.matrix(name = 'inputs') 

        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)

        if not theano_rng:  
            theano_rng = RandomStreams(1)

        # SET UP VARIABLES AND PARAMETERS 
        wdfh_init = numpy_rng.normal(size=(numvisD, numvisY*2, numhid)).astype(theano.config.floatX)*0.01
        self.wdfh = theano.shared(value = wdfh_init, name = 'wdfh')
        self.bvisD = theano.shared(value = numpy.zeros(numvisD, dtype=theano.config.floatX), name='bvisD')
        self.bvisY = theano.shared(value = numpy.zeros(numvisY*2, dtype=theano.config.floatX), name='bvisY')
        self.bhid = theano.shared(value = 0.0*numpy.ones(numhid, dtype=theano.config.floatX), name='bhid')
        self.params = [self.wdfh, self.bhid, self.bvisD, self.bvisY]

        # DEFINE THE LAYER FUNCTION 
        self.inputsD = self.inputs[:, :numvisD]
        self.inputsY = self.inputs[:, numvisD:]
        self.inputsY1 = self.inputsY[:, :numvisY]
        self.inputsY2 = self.inputsY[:, numvisY:]
        if self.corruption_level > 0.0: 
            if self.corruption_type=='zeromask':
                self._corruptedD = theano_rng.binomial(size=self.inputsD.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsD
                self._corruptedY1 = theano_rng.binomial(size=self.inputsY1.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsY1
                self._corruptedY2 = theano_rng.binomial(size=self.inputsY2.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsY2
            elif self.corruption_type=='gaussian':
                self._corruptedD = theano_rng.normal(size=self.inputsD.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsD
                self._corruptedY1 = theano_rng.normal(size=self.inputsY1.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsY1
                self._corruptedY2 = theano_rng.normal(size=self.inputsY2.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsY2
            elif self.corruption_type=='none':
                self._corruptedD = self.inputsD
                self._corruptedY1 = self.inputsY1
                self._corruptedY2 = self.inputsY2
            else:
                assert False, "unknown corruption type"
        else:
            self._corruptedD = self.inputsD
            self._corruptedY1 = self.inputsY1
            self._corruptedY2 = self.inputsY2

        self._uncorruptedD = self.inputsD
        self._uncorruptedY1 = self.inputsY1
        self._uncorruptedY2 = self.inputsY2

        self._featuresD = self._corruptedD   # we may eventually decide eventually to learn features on fixations, too
        self._featuresY = T.concatenate((self._corruptedY1, self._corruptedY2), axis=1)

        self._modweightsD = T.tensordot(self._featuresD, self.wdfh, axes=([1], [0]))
        self._modweightsY = T.tensordot(self._featuresY, self.wdfh, axes=([1], [1]))
        self._hiddens = T.nnet.sigmoid(T.sum(self._featuresY.dimshuffle(0,1,'x') * self._modweightsD, 1) + self.bhid)
        self._reconsY = T.sum(self._hiddens.dimshuffle(0,'x',1) * self._modweightsD, 2) + self.bvisY
        self._reconsY1 = self._reconsY[:, :self.numvisY]
        self._reconsY2 = self._reconsY[:, self.numvisY:]
        self._reconsD = T.sum(self._hiddens.dimshuffle(0,'x',1) * self._modweightsY, 2) + self.bvisD
        self._reconsD = T.nnet.sigmoid(self._reconsD)  #fixation deltas assumed binary 
        if self.output_type == 'binary':
            self._reconsY = T.nnet.sigmoid(self._reconsY)
        elif self.output_type == 'real':
            pass 
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        self._uncorruptedfeaturesD = self._uncorruptedD   # may decide to learn features on fixations, too
        self._uncorruptedfeaturesY = T.concatenate((self._uncorruptedY1, self._uncorruptedY2), axis=1)
        self._uncorruptedmodweightsD = T.tensordot(self._uncorruptedfeaturesD, self.wdfh, axes=([1], [0]))
        self._uncorruptedmodweightsY = T.tensordot(self._uncorruptedfeaturesY, self.wdfh, axes=([1], [1]))
        self._uncorruptedhiddens = T.nnet.sigmoid(T.sum(self._uncorruptedfeaturesY.dimshuffle(0,1,'x') * self._uncorruptedmodweightsD, 1) + self.bhid)
        self._uncorruptedreconsY = T.sum(self._uncorruptedhiddens.dimshuffle(0,'x',1) * self._uncorruptedmodweightsD, 2) + self.bvisY
        self._uncorruptedreconsD = T.sum(self._uncorruptedhiddens.dimshuffle(0,'x',1) * self._uncorruptedmodweightsY, 2) + self.bvisD
        self._uncorruptedreconsD = T.nnet.sigmoid(self._uncorruptedreconsD)  #fixation deltas assumed binary 
        if self.output_type == 'binary':
            self._uncorruptedreconsY = T.nnet.sigmoid(self._uncorruptedreconsY)
        elif self.output_type == 'real':
            pass 
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

 
        # DEFINE WEIGHTCOST
        self._weightcost = self.weightcost * (self.wdfh**2).sum() 
        if self.contraction > 0.0:
            self._contraction_cost = T.sum( ((self._hiddens * (1 - self._hiddens))**2) * T.sum(T.sum(self.wdfh**2, axis=1), axis=0).dimshuffle('x', 0), axis=1)

        # ATTACH COST FUNCTIONS
        self._costpercaseD = -T.sum(0.5* (self.inputsD*T.log(self._reconsD)+(1.0-self.inputsD)*T.log(1.0-self._reconsD)), axis=1) 
        if self.output_type == 'binary':
            self._costpercaseY = - T.sum(0.5* (self.inputsY*T.log(self._reconsY) + (1.0-self.inputsY)*T.log(1.0-self._reconsY)), axis=1) 
        elif self.output_type == 'real':
            self._costpercaseY = T.sum(0.5*((self.inputsY-self._reconsY)**2), axis=1)
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"
        if self.contraction > 0.0:
            self._costpercase = self._costpercaseY + self._costpercaseD + self.contraction * self._contraction_cost
        else:
            self._costpercase = self._costpercaseY + self._costpercaseD 
        self._weightcost = self._weightcost
        self._cost = T.mean(self._costpercase) + self._weightcost 
        self._grads = T.grad(self._cost, self.params)

        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER 
        self.uncorruptedfeaturesY = theano.function([self.inputs],self._uncorruptedfeaturesY)
        self.uncorruptedmodweightsD = theano.function([self.inputs],self._uncorruptedmodweightsD)
        self.uncorruptedmodweightsY = theano.function([self.inputs],self._uncorruptedmodweightsY)
        self.uncorruptedhiddens = theano.function([self.inputs],self._uncorruptedhiddens)
        self.uncorruptedreconsY = theano.function([self.inputs],self._uncorruptedreconsY)
        self.uncorruptedreconsD = theano.function([self.inputs],self._uncorruptedreconsD)
        self.reconsD = theano.function([self.inputs], self._reconsD)
        self.reconsY = theano.function([self.inputs], self._reconsY)
        self.cost = theano.function([self.inputs], self._cost)
        self.grads = theano.function([self.inputs], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))


    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])



