"""

Classes that define how vector spaces are formatted

Most of our models can be viewed as linearly transforming
one vector space to another. These classes define how the
vector spaces should be represented as theano/numpy
variables.

For example, the VectorSpace class just represents a
vector space with a vector, and the model can transform
between spaces with a matrix multiply. The Conv2DSpace
represents a vector space as an image, and the model
can transform between spaces with a 2D convolution.

To make models as general as possible, models should be
written in terms of Spaces, rather than in terms of
numbers of hidden units, etc. The model should also be
written to transform between spaces using a generic
linear transformer from the pylearn2.linear module.

The Space class is needed so that the model can specify
what kinds of inputs it needs and what kinds of outputs
it will produce when communicating with other parts of
the library. The model also uses Space objects internally
to allocate parameters like hidden unit bias terms in
the right space.

"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import theano.tensor as T
import theano.sparse
from theano.tensor import TensorType
from theano import config
import functools
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.space import Space, VectorSpace

if theano.sparse.enable_sparse:
    # We know scipy.sparse is available
    import scipy.sparse


class Conv3DSpace(Space):
    """A space whose points are defined as (multi-channel) images."""
    def __init__(self, shape, sequence_length, channels = None, num_channels = None, axes = None):
        """
        Initialize a Conv2DSpace.

        Parameters
        ----------
        shape : sequence, length 3
            The shape of a sequence of images, i.e. (time, rows, cols).
        sequence_length: int
            Length of the third dimension (frames, time,  etc)
        num_channels: int     (synonym: channels)
            Number of channels in the image, i.e. 3 if RGB.
        axes: A tuple indicating the semantics of each axis.
                'b' : this axis is the batch index of a minibatch.
                'c' : this axis the channel index of a minibatch.
                <i>  : this is topological axis i (i.e., 0 for rows,
                                  1 for cols)

                For example, a PIL image has axes (0, 1, 'c') or (0, 1).
                The pylearn2 image displaying functionality uses
                    ('b', 0, 1, 'c') for batches and (0, 1, 'c') for images.
                theano's conv2d operator uses ('b', 'c', 0, 1) images.
        """

        assert (channels is None) + (num_channels is None) == 1
        if num_channels is None:
            num_channels = channels

        assert isinstance(num_channels, py_integer_types)

        if not hasattr(shape, '__len__') or len(shape) != 2:
            raise ValueError("shape argument to Conv3DSpace must be length 3")
        assert all(isinstance(elem, py_integer_types) for elem in shape)
        assert all(elem > 0 for elem in shape)
        assert isinstance(num_channels, py_integer_types)
        assert num_channels > 0
        # Convert shape to a tuple, so it can be hashable, and self can be too
        self.shape = tuple(shape)
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        if axes is None:
            axes = ('b', 't', 'c', 0, 1)
        assert len(axes) == 5
        self.axes = tuple(axes)

    def __str__(self):
        return "Conv3DSpace{shape=%s,sequence_length=%d,num_channels=%d}" % \
                (str(self.shape), self.sequence_length, self.num_channels)

    def __eq__(self, other):
        return type(self) == type(other) and \
                self.shape == other.shape and \
                self.num_channels == other.num_channels \
                and self.sequence_length == other.sequence_length \
                and tuple(self.axes) == tuple(other.axes)

    def __hash__(self):
        return hash((type(self), self.shape, self.sequence_length,
                        self.num_channels, self.axes))

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        dims = { 't' : self.sequence_length, 0: self.shape[0], 1: self.shape[1], 'c' : self.num_channels }
        shape = [ dims[elem] for elem in self.axes if elem != 'b' ]
        return np.zeros(shape)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        if not isinstance(n, py_integer_types):
            raise TypeError("Conv3DSpace.get_origin_batch expects an int, got " +
                    str(n) + " of type " + str(type(n)))
        assert n > 0
        dims = { 'b' : n, 't': self.sequence_length, 0: self.shape[0], 1: self.shape[1], 'c' : self.num_channels }
        shape = [ dims[elem] for elem in self.axes ]
        return np.zeros(shape)

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if dtype is None:
            dtype = config.floatX

        broadcastable = [False] * 5
        broadcastable[self.axes.index('c')] = (self.num_channels == 1)
        broadcastable[self.axes.index('b')] = (batch_size == 1)
        broadcastable[self.axes.index('t')] = (self.sequence_length == 1)
        broadcastable = tuple(broadcastable)

        rval = TensorType(dtype=dtype,
                          broadcastable=broadcastable
                         )(name=name)
        if config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(n=n)
        return rval

    @functools.wraps(Space.batch_size)
    def batch_size(self, batch):
        self.validate(batch)
        return batch.shape[self.axes.index('b')]

    @functools.wraps(Space.np_batch_size)
    def np_batch_size(self, batch):
        self.np_validate(batch)
        return batch.shape[self.axes.index('b')]

    @staticmethod
    def convert(tensor, src_axes, dst_axes):
        """
            tensor: a 4 tensor representing a batch of images

            src_axes: the axis semantics of tensor

            Returns a view of tensor using the axis semantics defined
            by dst_axes. (If src_axes matches dst_axes, returns
            tensor itself)

            Useful for transferring tensors between different
            Conv2DSpaces.
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 5
        assert len(dst_axes) == 5

        if src_axes == dst_axes:
            return tensor

        shuffle = [ src_axes.index(elem) for elem in dst_axes ]

        return tensor.dimshuffle(*shuffle)

    @staticmethod
    def convert_numpy(tensor, src_axes, dst_axes):
        """
            tensor: a 5D tensor representing a batch of images

            src_axes: the axis semantics of tensor

            Returns a view of tensor using the axis semantics defined
            by dst_axes. (If src_axes matches dst_axes, returns
            tensor itself)

            Useful for transferring tensors between different
            Conv2DSpaces.
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 5
        assert len(dst_axes) == 5

        if src_axes == dst_axes:
            return tensor

        shuffle = [ src_axes.index(elem) for elem in dst_axes ]

        return tensor.transpose(*shuffle)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):

        # Patch old pickle files
        if not hasattr(self, 'num_channels'):
            self.num_channels = self.nchannels

        return self.shape[0] * self.shape[1] * self.sequence_length * self.num_channels

    @functools.wraps(Space.validate)
    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("Conv3DSpace batches must be theano Variables, got "+str(type(batch)))
        if not isinstance(batch.type, (theano.tensor.TensorType,CudaNdarrayType)):
            raise TypeError()
        if batch.ndim != 5:
            raise ValueError()
        for val in get_debug_values(batch):
            self.np_validate(val)

    @functools.wraps(Space.np_validate)
    def np_validate(self, batch):
        if (not isinstance(batch, np.ndarray)
                and type(batch) != 'CudaNdarray'):
            raise TypeError("The value of a Conv3DSpace batch should be a "
                    "numpy.ndarray, or CudaNdarray, but is %s."
                    % str(type(batch)))
        if batch.ndim != 5:
            raise ValueError("The value of a Conv3DSpace batch must be "
                    "5D, got %d dimensions for %s." % (batch.ndim, batch))

        d = self.axes.index('c')
        actual_channels = batch.shape[d]
        if actual_channels != self.num_channels:
            raise ValueError("Expected axis %d to be number of channels (%d) "
                    "but it is %d" % (d, self.num_channels, actual_channels))
        assert batch.shape[self.axes.index('c')] == self.num_channels

        for coord in [0, 1]:
            d = self.axes.index(coord)
            actual_shape = batch.shape[d]
            expected_shape = self.shape[coord]
            if actual_shape != expected_shape:
                raise ValueError("Conv3DSpace with shape %s and axes %s "
                        "expected dimension %s of a batch (%s) to have "
                        "length %s but it has %s"
                        % (str(self.shape), str(self.axes), str(d), str(batch),
                           str(expected_shape), str(actual_shape)))

    @functools.wraps(Space.np_format_as)
    def np_format_as(self, batch, space):
        self.np_validate(batch)
        if isinstance(space, VectorSpace):
            if self.axes[0] != 'b':
                # We need to ensure that the batch index goes on the first axis before the reshape
                new_axes = ['b'] + [axis for axis in self.axes if axis != 'b']
                batch = batch.transpose(*[self.axes.index(axis) for axis in new_axes])
            return batch.reshape((batch.shape[0], self.get_total_dimension()))
        if isinstance(space, Conv3DSpace):
            return Conv3DSpace.convert_numpy(batch, self.axes, space.axes)
        raise NotImplementedError("Conv3DSPace doesn't know how to format as "+str(type(space)))

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        self.validate(batch)
        if isinstance(space, VectorSpace):
            if self.axes[0] != 'b':
                # We need to ensure that the batch index goes on the first axis before the reshape
                new_axes = ['b'] + [axis for axis in self.axes if axis != 'b']
                batch = batch.transpose(*[self.axes.index(axis) for axis in new_axes])
            return batch.reshape((batch.shape[0], self.get_total_dimension()))
        if isinstance(space, Conv3DSpace):
            return Conv3DSpace.convert(batch, self.axes, space.axes)
        raise NotImplementedError("Conv3DSPace doesn't know how to format as "+str(type(space)))



