from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
from pylearn2.models.mlp import Layer
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
import functools
from pylearn2.utils import sharedX
import theano.tensor as T
from theano import config
import numpy as np
from theano import function
from theano.gof.op import get_debug_values
from theano.printing import Print
import theano

from collections import OrderedDict


def batched_softmax(x):
    """
    :param x: A Tensor3
    This function computes the batched softmax
    """
                
    result, updates = theano.scan(fn=lambda x_mat:
            T.nnet.softmax(x_mat),
            outputs_info=None,
            sequences=[x],
            non_sequences=None)
    return result
                
class MatrixSpace(Space):
    """A space whose points are defined as fixed-length vectors."""
    def __init__(self, num_row, num_column):
        """
        Initialize a MatrixSpace.

        Parameters
        ----------
        dim : int
            Dimensionality of a vector in this space.
        sparse: bool
            Sparse vector or not
        """
        self.num_row = num_row
        self.num_column = num_column

    def __str__(self):
        return self.__class__.__name__

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return np.zeros((self.num_row,self.num_column))

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        return np.zeros((n, self.num_row, self.num_column))

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if dtype is None:
            dtype = config.floatX

        return T.tensor3(name=name, dtype=dtype)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.num_column * self.num_row

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        raise NotImplementedError()
        if isinstance(space, CompositeSpace):
            pos = 0
            pieces = []
            for component in space.components:
                width = component.get_total_dimension()
                subtensor = batch[:,pos:pos+width]
                pos += width
                formatted = VectorSpace(width).format_as(subtensor, component)
                pieces.append(formatted)
            return tuple(pieces)

        if isinstance(space, Conv2DSpace):
            if space.axes[0] != 'b':
                raise NotImplementedError("Will need to reshape to ('b',*) then do a dimshuffle. Be sure to make this the inverse of space._format_as(x, self)")
            dims = { 'b' : batch.shape[0], 'c' : space.num_channels, 0 : space.shape[0], 1 : space.shape[1] }

            shape = tuple( [ dims[elem] for elem in space.axes ] )

            rval = batch.reshape(shape)

            return rval

        raise NotImplementedError("VectorSpace doesn't know how to format as "+str(type(space)))

    def __eq__(self, other):
        return type(self) == type(other) and self.num_row == other.num_row and self.num_column == other.num_column

    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("MatrixSpace batch should be a theano Variable, got "+str(type(batch)))
        if batch.ndim != 3:
            raise ValueError('MatrixSpace batches must be 2D, got %d dimensions' % batch.ndim)

    def np_validate(self, batch):
        pass #ALL IS FINE IN THE BEST OF WORLDS! XXX
        #return self.validate(batch)

    def np_batch_size(self, X):
        return len(X)

class MultiSoftmax(Layer):

    def __init__(self, n_groups, n_classes, layer_name, irange = None,
                 istdev = None, sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False, max_col_norm = None):
        """
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self

        assert isinstance(n_classes, py_integer_types)

        self.output_space = MatrixSpace(n_groups, n_classes)
        self.b = sharedX( np.zeros((n_groups, n_classes,)), name = 'softmax_b')

    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    def get_monitoring_channels(self):
        return OrderedDict()

    def get_monitoring_channels_from_state(self, state, target=None):
        return OrderedDict()
        
    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_groups,self.n_classes))
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.input_dim,self.n_groups,self.n_classes) * self.istdev
        else:
            raise NotImplementedError()

        self.W = sharedX(W,  'softmax_W' )

        self._params = [ self.b, self.W ]

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt, self.input_space.axes, ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        assert self.W.ndim == 3

        Z = T.tensordot(state_below, self.W, axes=[[1],[0]]) + self.b

        rval = batched_softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=2).mean()

    def cost_matrix(self, Y, Y_hat):
        return -Y * T.log(Y_hat+0.000001)

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()

    def censor_updates(self, updates):
        return
        if self.max_row_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')
        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))
