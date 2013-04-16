"""
 A bi-directional RNN.

TODO:
    - ensure compatibility with pylearn2

maintainer: Razvan (r.pascanu@gmail)
"""

import numpy
import theano
import theano.tensor as TT
from utils import safe_clone

class biRNN(object):
    def __init__(self,
                 nhids =50,
                 nouts = 8,
                 nins = 2,
                 activ = TT.nnet.sigmoid,
                 seed = 234,
                 bs = 16,
                ):
        # 0. Keep track of arguments
        self.bs = bs
        self.nhids = nhids
        self.nouts = nouts
        self.nins = nins
        self.activ = activ
        self.seed = seed
        self.bs = bs
        floatX = theano.config.floatX
        self.rng = numpy.random.RandomState(seed)

        # 1. Generating Theano variables
        # DenseSequence space
        self.x = TT.tensor3('x')
        # IndexSequence space
        self.t = TT.vector('t')
        self.inputs = [self.x, self.y]
        self.W_uhf = numpy.asarray(
            self.rng.normal(size=(self.nin, self.nhids), loc=0, scale=.01),
            dtype=floatX)
        self.W_uhb = numpy.asarray(
            self.rng.normal(size=(self.nin, self.nhids), loc=0, scale=.01),
            dtype=floatX)
        self.W_hhf = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nhids), loc=0, scale=1),
            dtype=floatX)
        self.W_hhb = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nhids), loc=0, scale=1),
            dtype=floatX)
        slef.W_hyf = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nouts), loc=0, sclae=.1),
            dtype=floatX)
        slef.W_hyb = numpy.asarray(
            self.rng.normal(size=(self.nhids, self.nouts), loc=0, sclae=.1),
            dtype=floatX)
        # sparsifying hidden weights
        for dx in xrange(self.nhids):
            psng = self.rng.permutation(n_hidden)
            self.W_hhf[dx][spng[15:]] = 0.
            psng = self.rng.permutation(n_hidden)
            self.W_hhb[dx][spng[15:]] = 0.
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
        # Do I need to store best params !?
        self.best_params = [(x.name, x.get_value()) for x in self.params]
        self.params_shape = [x.get_value(borrow=True).shape for x in
                             self.params]

        # 2. Constructing Theano graph
        h0_f = TT.alloc(numpy.array(0,dtype=floatX), self.bs,
                              self.nhids)
        h0_b = TT.alloc(numpy.array(0, dtype=floatX), self.bs,
                               self.nhids)

        # Do we use to much memory!?
        p_hf = TT.dot(self.x.flatten(2), self.W_uhf) + self.b_hhf
        p_hb = TT.dot(self.x[::-1].flatten(2), self.W_uhb) + self.b_hhb

        def recurrent_fn(pf_t, pb_t, hf_tm1, hb_tm1):
            hf_t = activ(TT.dot(hf_tm1, self.W_hhf) + pf_t)
            hb_t = activ(TT.dot(hb_tm1, self.W_hhb) + pb_t)
            return hf_t, hb_t
        # provide sequence length !? is better on GPU
        [h_f, h_b], _ = theano.scan(
            recurrent_fn,
            sequences = [
                p_hf.reshape((-1, self.bs, self.nhids)),
                p_hf.reshape((-1, self.bs, self.nhids))]
            outputs_info = [h0_f, h0_b],
            name = 'bi-RNN',
            mode = theano.Mode(linker='cvm'),
            profile = 0)
        h_b = h_b[::-1]
        y = TT.nnet.softmax(
            TT.dot(h_f.flatten(2), self.W_hyf) +
            TT.dot(h_b.flatten(2), self.W_hyb) +
            self.b_hy)
        my = y.reshape((-1, self.bs, self.nouts)).max(axis=0)
        nll = -TT.log(my[TT.constant(numpy.arange(self.bs)), self.t])
        self.train_cost = nll.mean()
        self.Gyvs = lambda *args:\
            TT.Lop(y, self.params,
                   TT.Rop(y, self.params, args) /\
                   ((1-y)*y*self.bs))
        pred = TT.argmax(my, axis=1)
        self.error = TT.mean(TT.neq(pred, self.t)) * 100

        # Graphs for validation purposes
        vt = TT.scalar('vt')
        vh0_f = TT.alloc(numpy.array(0,dtype=floatX), 1,
                              self.nhids)
        vh0_b = TT.alloc(numpy.array(0, dtype=floatX), 1,
                               self.nhids)
        vy = safe_clone(y, replace={h0_f:vh0_f,
                                      h0_b:vh0_b})
        my = TT.neq(vy.max(axis=0),argmax(), vt)
        self.validate = theano.function([self.x, vt], my,
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
