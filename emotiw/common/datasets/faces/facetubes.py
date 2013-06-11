"""
This files defines a Pylearn2 Dataset holding facetubes.
"""
# Basic Python packages
import functools

# External dependencies
import numpy as np

# In-house dependencies
import theano
from theano import config
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.tensor import TensorType

from pylearn2.datasets import Dataset
from pylearn2.utils.iteration import (FiniteDatasetIterator,
                                      resolve_iterator_class)
from pylearn2.space import VectorSpace, Space

# Current project


class FaceTubeSpace(Space):
    """Space for variable-length sequences of images.

    All the images in all sequences have the same dimensions and number
    of channels, but the length of different sequences may be different.
    Hence, this space is restricted to batch sizes of 1.
    """
    def __init__(self, shape, num_channels, axes=None):
        if axes is None:
            # 'b' indicates batch size, should always be 1
            # 't' indicates time steps
            # 0 is the image height,
            # 1 is the image width,
            # 'c' is the channel (for instance, R G or B)
            axes = ('b', 't', 0, 1, 'c')

        self.shape = tuple(shape)
        self.num_channels = num_channels
        self.axes = tuple(axes)

    def __str__(self):
        return 'FaceTubeSpace{shape=%s,num_channels=%s}' % (
                str(self.shape), str(self.num_channels))

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape and
                self.num_channels == other.num_channels and
                self.axes == other.axes)

    def __hash__(self):
        return hash((type(self), self.shape, self.num_channels, self.axes))

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        # the length of 't' will vary among examples, we use 1 here,
        # but it could change
        dims = {0: self.shape[0],
                1: self.shape[1],
                'c': self.num_channels,
                't': 1}
        shape = [dims[elem] for elem in self.axes if elem != 'b']
        return np.zeros(shape)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        assert n == 1, "batch processing not supported for face tubes"
        # the length of 't' will vary among examples, we use 1 here,
        # but it could change
        dims = {0: self.shape[0],
                1: self.shape[1],
                'c': self.num_channels,
                't': 1,
                'b': 1}
        shape = [dims[elem] for elem in self.axes]
        return np.zeros(shape)

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if dtype is None:
            dtype = config.floatX

        broadcastable = [False] * 5
        broadcastable[self.axes.index('c')] = (self.num_channels == 1)
        broadcastable[self.axes.index('b')] = True
        broadcastable = tuple(broadcastable)

        rval = TensorType(dtype=dtype,
                          broadcastable=broadcastable)(name=name)
        if config.compute_test_value != 'off':
            rval.tag.test_value = self.get_origin_batch(n=1)

    @functools.wraps(Space.batch_size)
    def batch_size(self, batch):
        self.validate(batch)
        return 1

    @functools.wraps(Space.np_batch_size)
    def np_batch_size(self, batch):
        self.np_validate(batch)
        return 1

    @functools.wraps(Space.validate)
    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("%s batches must be Theano Variables, got %s"
                            % (str(type(self)), str(type(batch))))
        if not isinstance(batch.type, (theano.tensor.TensorType,
                                       CudaNdarrayType)):
            raise TypeError()
        if batch.ndim != 5:
            raise ValueError()
        if not batch.broadcastable[self.axes.index('b')]:
            raise ValueError("%s batches should be broadcastable along the "
                             "'b' (batch size) dimension." % str(type(self)))
        for val in get_debug_values(batch):
            self.np_validate(val)

    @functools.wraps(Space.np_validate)
    def np_validate(self, batch):
        if (not isinstance(batch, np.ndarray)
                and type(batch) != 'CudaNdarray'):
            raise TypeError("The value of a %s batch should be a "
                            "numpy.ndarray, or CudaNdarray, but is %s."
                            % (str(type(self)), str(type(batch))))
        if batch.ndim != 5:
            raise ValueError("The value of a %s batch must be "
                             "5D, got %d dimensions for %s."
                             % (str(type(self)), batch.ndim, batch))

        d = self.axes.index('c')
        actual_channels = batch.shape[d]
        if actual_channels != self.num_channels:
            raise ValueError("Expected axis %d to be number of channels (%d) "
                             "but it is %d"
                             % (d, self.num_channels, actual_channels))
        assert batch.shape[self.axes.index('c')] == self.num_channels

        assert batch.shape[self.axes.index('b')] == 1

        for coord in [0, 1]:
            d = self.axes.index(coord)
            actual_shape = batch.shape[d]
            expected_shape = self.shape[coord]
            if actual_shape != expected_shape:
                raise ValueError(
                    "%s with shape %s and axes %s "
                    "expected dimension %s of a batch (%s) to have "
                    "length %s but it has %s"
                    % (str(type(self)), str(self.shape), str(self.axes),
                       str(d), str(batch), str(expected_shape),
                       str(actual_shape)))

    @functools.wraps(Space.np_format_as)
    def np_format_as(self, batch, space):
        self.np_validate(batch)

        if isinstance(space, FaceTubeSpace):
            assert len(self.axes) == 5
            assert len(space.axes) == 5
            if self.axes == space.axes:
                return batch
            new_axes = [self.axes.index(e) for e in space.axes]
            return batch.transpose(*new_axes)

        if isinstance(space, VectorSpace):
            # space.dim has to have the right size for current batch,
            # or be None
            prod_batch_shape = np.prod(batch.shape)
            if space.dim not in (None, prod_batch_shape):
                raise TypeError(
                    "%s cannot convert to a VectorSpace of a "
                    "different size (space.dim=%s, should be None or %s)"
                    % (str(type(self)), space.dim, prod_batch_shape))
            if self.axes[0] != 'b':
                # We need to ensure that the batch index goes on the first axis
                # before the reshape
                new_axes = ['b'] + [axis for axis in self.axes if axis != 'b']
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in new_axes])
            return batch.reshape((batch.shape[0], -1))

        raise NotImplementedError("%s doesn't know how to format as %s"
                                  % (str(type(self)), str(type(space))))

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        self.validate(batch)

        if isinstance(space, FaceTubeSpace):
            assert len(self.axes) == 5
            assert len(space.axes) == 5
            if self.axes == space.axes:
                return batch
            new_axes = [self.axes.index(e) for e in space.axes]
            return batch.transpose(*new_axes)

        if isinstance(space, VectorSpace):
            if self.axes[0] != 'b':
                # We need to ensure that the batch index goes on the first axis
                # before the reshape
                new_axes = ['b'] + [axis for axis in self.axes if axis != 'b']
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in new_axes])
            return batch.reshape((batch.shape[0], -1))

        raise NotImplementedError("%s doesn't know how to format as %s"
                                  % (str(type(self)), str(type(space))))


class FaceTubeDataset(Dataset):
    def get_data(self):
        return self.data

    def get_data_specs(self):
        return self.data_specs

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            raise ValueError('iteration mode not provided and no default '
                             'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        if data_specs is None:
            data_specs = getattr(self, '_iter_data_specs', None)

        return FiniteDatasetIterator(
                self,
                mode(self.n_samples,
                     batch_size,
                     num_batches,
                     rng),
                data_specs=data_specs,
                return_tuple=return_tuple)
