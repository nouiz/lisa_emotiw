import numpy
import theano
import theano.tensor as T
from emotiw.common.aggregation import pooling

def test_max_pooling():
    x = T.tensor3()
    xval = numpy.random.rand(4,3,2)
    fn = theano.function([x], pooling.max_pool(x))
    fn(xval)

def test_mean_pooling():
    x = T.tensor3()
    xval = numpy.random.rand(4,3,2)
    fn = theano.function([x], pooling.mean_pool(x))
    fn(xval)

def test_pnorm_pooling():
    x = T.tensor3()
    p = 2.5
    fn = theano.function([x], pooling.pnorm_pool(x, p=p))
    xval = numpy.random.rand(4,3,2)
    rval = fn(xval)
    for i in xrange(xval.shape[1]):
        for j in xrange(xval.shape[2]):
            numpy.testing.assert_approx_equal(
                    rval[i,j],
                    numpy.sum(xval[:,i,j]**p)**(1./p))

def test_noisy_or_pooling():
    x = T.tensor3()
    p = 2.5
    fn = theano.function([x], pooling.noisy_or_pooling(x))
    xval = numpy.random.rand(4,3,2)
    rval = fn(xval)
    for i in xrange(xval.shape[1]):
        for j in xrange(xval.shape[2]):
            numpy.testing.assert_approx_equal(
                    rval[i,j],
                    1 - numpy.prod(1 - xval[:,i,j]))


def test_spatial_pyramid():
    x = T.tensor3()
    fn = theano.function([x], pooling.spatial_pyramid(x, scales=[1,2]))
    xval = numpy.random.rand(4,3,2)
    rval = fn(xval)
    numpy.testing.assert_array_almost_equal(rval[0], xval.max(axis=0))
    numpy.testing.assert_array_almost_equal(rval[1], xval[:2].max(axis=0))
    numpy.testing.assert_array_almost_equal(rval[2], xval[2:].max(axis=0))
