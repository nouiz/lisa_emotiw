import numpy, pylab
import cPickle
import warnings

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import conv


class Inference(object):

    # Class for inference of feature vectors given the input video blocks
    # based on the SAE model from the paper Konda, Kishore Reddy, 
    # Roland Memisevic, and Vincent Michalski. "The role of spatio-temporal 
    # synchrony in the encoding of motion." arXiv preprint arXiv:1306.3162 (2013).


    def __init__(self, numpy_rng, theano_rng = None, nvis=2560, npro=300):

        # @params     
        # numpy_rng   : number random generator used to generate weights
        # theano_rng  : Theano random generator; if None is given one is
        #               generated based on a seed drawn from `numpy_rng`
        # nvis        : dimensionality of input
        # npro        : number of product units 
        #               (dimensionality of output feature vectors)


        self.nvis  = nvis
        self.npro = npro
        self.theano_rng = theano_rng

        # Theano shared variables for computing mean 
        # and standard deviation (for contrast normalization)

        self.m0 = theano.shared(numpy.zeros((nvis,),dtype=theano.config.floatX),name='m0')
        self.s0 = theano.shared(numpy.ones((nvis,),dtype=theano.config.floatX),name='s0')
        

        if not theano_rng : 
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        # Theano shared variable for loading weight matrix
        # a product of filters weights and whitening matrix
        # comuted during training phase

        self.W = theano.shared(numpy.asarray(numpy_rng.uniform(
                        low=-0.01,
                        high=0.01,
                        size=(nvis,npro)), dtype=theano.config.floatX),name ='W')
      
        # Theano variable for minibatch of input samples. 
        # A matrix with each row being an input sample. 

        self.input = T.matrix(name = 'input')

        # Parameter list to be loaded using 'self.load' function
        self.params = [self.W, self.m0, self.s0]

        # Contrast normalization by mean centering followed by data centering

        m1 = self.input.mean(1)
        input_p = self.input - m1.dimshuffle(0, 'x')
        
        s1 = input_p.std(1) + input_p.std(1).mean() + 0.001
        input_p = input_p / s1.dimshuffle(0, 'x')
        
        input_p = input_p - self.m0.dimshuffle('x',0)
        input_p = input_p / self.s0.dimshuffle('x',0)

        # computation of factors

        self.factors = self.get_factors(input_p)    

        # computation of products (feature vectors)

        self.product = T.nnet.sigmoid(self.factors**2)
      
        # theano function returns self.products given input

        self.get_product = theano.function([self.input], self.product)

    
    def get_factors(self, input):
        # Function for applying combined whitenening 
        # matrix and filter weights to compute factors
        # @params:
        # input  : contrast normalized input.

        factors = T.dot(input,self.W)

        return factors

    def products_batchwise(self, input, batchsize):
        # Function to compute feature vectors batchwise. 
        # Not all input samples are loaded on to memory at once.
        # @params   :
        # input     : Matrix with each row a input sample.
        # batchsize : Size of minibatch of input samples.

        numbatches = input.shape[0] / batchsize
        mappings = numpy.zeros((input.shape[0], self.npro), dtype=theano.config.floatX)
        for batch in range(numbatches):
            mappings[batch*batchsize:(batch+1)*batchsize, :]=self.get_product(input[batch*batchsize:(batch+1)*batchsize])
        if numpy.mod(input.shape[0], batchsize) > 0:
            mappings[numbatches*batchsize:, :]=self.get_product(input[numbatches*batchsize:])
        return mappings


    # Functions for loading and saving the parameters.

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

    def updateparams_fromdict(self, newparams):
        for p in self.params:
            p.set_value(newparams[p.name])

    def get_params_dict(self):
        return dict([(p.name, p.get_value()) for p in self.params])

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def save_npz(self, filename):
        numpy.savez(filename, **(self.get_params_dict()))

    def load(self, filename):
        new_params = None
        try:
            new_params = numpy.load(filename)
        except IOError, e:
            warnings.warn('''Parameter file could not be loaded with numpy.load()!
                          Is the filename correct?\n %s''' % (e, ))
        if type(new_params) == numpy.ndarray:
            print "loading npy file"
            self.updateparams(new_params)
        elif type(new_params) == numpy.lib.npyio.NpzFile:
            print "loading npz file"
            self.updateparams_fromdict(new_params)
        else:
            warnings.warn('''Parameter file loaded, but variable type not
                          recognized. Need npz or ndarray.''', Warning)




