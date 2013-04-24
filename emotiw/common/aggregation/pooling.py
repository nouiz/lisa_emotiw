import numpy
import theano
import theano.tensor as T
from theano.printing import Print
floatX = theano.config.floatX
npy_floatX = getattr(numpy, floatX)

DIM_TIME  = 0
DIM_BATCH = 1
DIM_LAB   = 2

def max_pool(x):
    """
    :type x: T.tensor3
    Symbolic 3-tensor representing the conditional classification probabilities.
    The 3 dimensions are: batch size, temporal, and target labels and should be accessed using
    the macros above.

    :rval: T.matrix
    Class prediction probabilities for each clip, having max-pooled over time.
    """
    assert x.ndim == 3
    return x.max(axis=DIM_TIME)

def mean_pool(x):
    """ See documentation for maxpool. """
    assert x.ndim == 3
    return x.mean(axis=DIM_TIME)

def pnorm_pool(x, p=2):
    """
    :param x: see max_pool
    :param p: integer, specifying type of norm to use.
    """
    assert x.ndim == 3
    pXinv = npy_floatX(1./p)
    return T.sum(x**p, axis=DIM_TIME)**(pXinv)

def noisy_or_pooling(x):
    assert x.ndim == 3
    return 1 - T.prod(1 - x, axis=DIM_TIME)

def spatial_pyramid(x, fn=max_pool, scales=None, **kwargs):
    """
    Performs pooling over various quadrants of the data and then aggregates the result.
    :param x: see max_pool
    :param fn: pointer to function having prototype `function(x, **kwargs)`
    :param scales: list of quadrants over which to perform pooling.
    e.g. scales=[1,2] will perform pooling over the entire sequence (jointly), then pool
    individually over the first and second half of the data. The return vector would then be of
    length 3.
    :param kwargs: arguments to pass to max_pool.
    """
    assert DIM_TIME == 0
    assert scales
    for scale in scales:
        assert isinstance(scale, int)

    def chunk_pool(idx, x, scale):
        assert idx.ndim == 0
        assert x.ndim == 3
        assert scale.ndim == 0
        rval = fn(x[idx : idx + x.shape[0] / scale], **kwargs)
        assert rval.ndim == 2
        return rval

    rval = T.shape_padleft(T.zeros_like(x[0]))

    for scale in scales:
        indices = T.arange(0, x.shape[0], x.shape[0] / scale)
        temp, updates = theano.scan(chunk_pool,
                sequences = [indices],
                outputs_info = [None],
                non_sequences = [x, T.constant(scale)])
        rval = T.join(0, rval, temp)

    return rval[1:]
