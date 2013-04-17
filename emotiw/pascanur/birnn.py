"""
 A bi-directional RNN.

TODO:
    - ensure compatibility with pylearn2

maintainer: Razvan (r.pascanu@gmail)
"""

import numpy
import theano
import theano.tensor as TT
# New interface of scan - gives more control to the user and helps avoiding
# certain shortcommings of scan
from theano.sandbox.scan import scan
from utils import safe_clone

class biRNN(object):
    def __init__(self,
                 nhids =50,
                 nouts = 8,
                 nins = 2,
                 activ = TT.nnet.sigmoid,
                 seed = 234,
                 bs = 16, # batchsize
                 seqlen = 3 # sequence length - fixed during training
                ):
        # 0. Keep track of arguments
        self.bs = bs
        self.nhids = nhids
        self.nouts = nouts
        self.nins = nins
        self.activ = activ
        self.seed = seed
        self.bs = bs
        self.seqlen = seqlen
        floatX = theano.config.floatX
        self.rng = numpy.random.RandomState(seed)

        # 1. Generating Theano variables
        # DenseSequence space
        # We store data as 3D tensor with (time, batch-size, nfeatures)
        self.x = TT.tensor3('x')
        # IndexSequence space
        # We store data as 1D tensor where each the dimension goes over the
        # batch size (i.e. target of each sequence in the batch)
        self.t = TT.ivector('t') # target index for each element of batchsize
        self.inputs = [self.x, self.t]
        # Naming convention for letters after the `_`:
        # u - input
        # h - hidden
        # y - output
        # f - forward
        # b - backwards

        self.W_uhf = numpy.asarray(
            self.rng.normal(size=(self.nins, self.nhids), loc=0, scale=.01),
            dtype=floatX)
        self.W_uhb = numpy.asarray(
            self.rng.normal(size=(self.nins, self.nhids), loc=0, scale=.01),
            dtype=floatX)
        self.W_hhf = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nhids), loc=0, scale=1),
            dtype=floatX)
        self.W_hhb = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nhids), loc=0, scale=1),
            dtype=floatX)
        self.W_hyf = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nouts), loc=0, scale=.1),
            dtype=floatX)
        self.W_hyb = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nouts), loc=0, scale=.1),
            dtype=floatX)
        # sparsifying hidden weights (Ilya&Martens formula == ESN style
        # init)
        for dx in xrange(self.nhids):
            psng = self.rng.permutation(nhids)
            self.W_hhf[dx][psng[15:]] = 0.
            psng = self.rng.permutation(nhids)
            self.W_hhb[dx][psng[15:]] = 0.

        # Any spectral radius larger than .9 smaller than 1.1 should be fine
        sr = numpy.max(abs(numpy.linalg.eigvals(self.W_hhf)))
        self.W_hhf = numpy.float32(.97*self.W_hhf/sr)
        sr = numpy.max(abs(numpy.linalg.eigvals(self.W_hhb)))
        self.W_hhb = numpy.float32(.97*self.W_hhb/sr)
        self.b_hhf = numpy.zeros((nhids,), dtype=floatX)
        self.b_hhb = numpy.zeros((nhids,), dtype=floatX)
        self.b_hy = numpy.zeros((nouts,), dtype=floatX)

        self.W_uhf = theano.shared(self.W_uhf, name='W_uhf')
        self.W_uhb = theano.shared(self.W_uhb, name='W_uhb')
        self.W_hhf = theano.shared(self.W_hhf, name='W_hhf')
        self.W_hhb = theano.shared(self.W_hhb, name='W_hhb')
        self.W_hyf = theano.shared(self.W_hyf, name='W_hyf')
        self.W_hyb = theano.shared(self.W_hyb, name='W_hyb')
        self.b_hhf = theano.shared(self.b_hhf, name='b_hhf')
        self.b_hhb = theano.shared(self.b_hhb, name='b_hhb')
        self.b_hy = theano.shared(self.b_hy, name='b_hy')

        self.params = [self.W_uhf, self.W_uhb, self.W_hhf, self.W_hhb,
                       self.W_hyf, self.W_hyb, self.b_hhf, self.b_hhb,
                       self.b_hy]
        self.best_params = [(x.name, x.get_value()) for x in self.params]
        self.params_shape = [x.get_value(borrow=True).shape for x in
                             self.params]

        # 2. Constructing Theano graph
        # Note: new interface of scan asks the user to provide a memory
        # buffer that contains the initial state but which is also used
        # internally by scan to store the intermediate values of its
        # computations - hence the initial state is a 3D tensor
        h0_f = TT.alloc(numpy.array(0,dtype=floatX), self.seqlen+1, self.bs,
                              self.nhids)
        h0_b = TT.alloc(numpy.array(0, dtype=floatX), self.seqlen+1, self.bs,
                               self.nhids)

        # Do we use to much memory!?
        p_hf = TT.dot(self.x.reshape((self.seqlen*self.bs, self.nins)), self.W_uhf) + self.b_hhf
        p_hb = TT.dot(self.x[::-1].reshape((self.seqlen*self.bs, self.nins)), self.W_uhb) + self.b_hhb

        def recurrent_fn(pf_t, pb_t, hf_tm1, hb_tm1):
            hf_t = activ(TT.dot(hf_tm1, self.W_hhf) + pf_t)
            hb_t = activ(TT.dot(hb_tm1, self.W_hhb) + pb_t)
            return hf_t, hb_t
        # provide sequence length !? is better on GPU
        [h_f, h_b], _ = scan(
            recurrent_fn,
            sequences = [
                p_hf.reshape((self.seqlen, self.bs, self.nhids)),
                p_hb.reshape((self.seqlen, self.bs, self.nhids))],
            states = [h0_f, h0_b],
            n_steps = self.seqlen,
            name = 'bi-RNN',
            profile = 0)
        h_b = h_b[::-1]
        # Optionally do the max over hidden layer !?
        # I'm afraid the semantics for RNN are somewhat different than MLP
        y = TT.nnet.softmax(
            TT.dot(h_f.reshape((self.seqlen * self.bs+self.bs, self.nhids)), self.W_hyf) + # Check doc flatten
            TT.dot(h_b.reshape((self.seqlen * self.bs+self.bs, self.nhids)), self.W_hyb) +
            self.b_hy)
        my = y.reshape((self.seqlen+1, self.bs, self.nouts)).max(axis=0)
        nll = -TT.log(
            my[TT.constant(numpy.arange(self.bs)), self.t])
        self.train_cost = nll.mean()
        self.error = TT.mean(TT.neq(my.argmax(axis=1), self.t) * 100.)
        ## |-----------------------------
        # - Computing metric times a vector efficiently for p(y|x)
        # Assume softmax .. we might want sigmoids though
        self.Gyvs = lambda *args:\
            TT.Lop(y, self.params,
                   TT.Rop(y, self.params, args) /\
                   (y*numpy.array(self.bs, dtype=floatX)))
        # Computing metric times a vector effciently for p(h|x)
        if activ == TT.nnet.sigmoid:
            fn = lambda x : (1-x)*x*numpy.array(self.bs, dtype=floatX)
        elif activ == TT.tanh:
            # Please check formula !!!! It is probably wrong
            fn = lambda x:(.5-x/2)*(x/2+.5)*numpy.array(self.bs,
                                                        dtype=floatX)
        else: # Assume linear or piece-wise linear activation
            fn = lambda x: numpy,array(self.bs, dtype=floatX)
        self.Ghfvs = lambda *args:\
                TT.Lop(h_f, self.params,
                       TT.Rop(h_f, self.params, args) / fn(h_f))
        self.Ghbvs = lambda *args:\
                TT.Lop(h_b, self.params,
                       TT.Rop(h_b, self.params, args) / fn(h_b))
        ## ------------------ |

        vx = TT.matrix('vx')
        vt = TT.iscalar('vt')
        vh0_f = TT.alloc(numpy.array(0,dtype=floatX), self.seqlen+1, self.nhids)
        vh0_b = TT.alloc(numpy.array(0, dtype=floatX), self.seqlen+1, self.nhids)

        # Do we use to much memory!?
        vp_hf = TT.dot(vx, self.W_uhf) + self.b_hhf
        vp_hb = TT.dot(vx[::-1], self.W_uhb) + self.b_hhb

        def recurrent_fn(pf_t, pb_t, hf_tm1, hb_tm1):
            hf_t = activ(TT.dot(hf_tm1, self.W_hhf) + pf_t)
            hb_t = activ(TT.dot(hb_tm1, self.W_hhb) + pb_t)
            return hf_t, hb_t
        # provide sequence length !? is better on GPU
        [vh_f, vh_b], _ = scan(
            recurrent_fn,
            sequences = [vp_hf, vp_hb],
            states = [vh0_f, vh0_b],
            name = 'valid bi-RNN',
            n_steps = vp_hf.shape[0],
            profile = 0)
        vh_b = vh_b[::-1]
        # Optionally do the max over hidden layer !?
        # I'm afraid the semantics for RNN are somewhat different than MLP
        vy = TT.nnet.softmax(
            TT.dot(vh_f, self.W_hyf) +
            TT.dot(vh_b, self.W_hyb) +
            self.b_hy)
        my = TT.neq(vy.max(axis=0).argmax(), vt)
        self.validate = theano.function([vx, vt], my,
                                        name='validation',
                                        profile=0)



    def save(self, filename):
        """
        Personally I don't like relying on pickling the class to save, but
        rather to saving explicitly to minimize the size of the saved file
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, vals)

    def load(self, filenma):
        values = numpy.load(filename)
        for param in self.params:
            param.set_value(values[param.name], borrow=True)
