from nose.plugins.skip import SkipTest
from contextlib import contextmanager
import numpy
import crop_match


@contextmanager
def skip_on_import_error():
    try:
        yield
    except ImportError:
        raise SkipTest


def test_python():
    yield check_invariance, crop_match._haar_like_py
    yield check_known_values, crop_match._haar_like_py
    yield check_matching, crop_match._match_subregion_py


def test_cython():
    with skip_on_import_error():
        import _crop_match
        yield check_invariance, _crop_match.haar_like
        yield check_known_values, _crop_match.haar_like
        yield check_matching, _crop_match.match_subregion


def check_invariance(haar_like):
    x = numpy.random.normal(size=(10, 10, 1))
    y = numpy.concatenate([numpy.concatenate([x, x], axis=1)] * 2, axis=0)
    int_y = crop_match._integral_image(y)
    ref = haar_like(crop_match._integral_image(x))
    numpy.testing.assert_allclose(ref, haar_like(int_y, 0, 0, 10, 10))
    numpy.testing.assert_allclose(ref, haar_like(int_y, 10, 0, None, 10))
    numpy.testing.assert_allclose(ref, haar_like(int_y, 0, 10, 10, None))
    numpy.testing.assert_allclose(ref, haar_like(int_y, 10, 10, None, None))


def check_known_values(haar_like):
    a = numpy.arange(16, dtype='float64').reshape((4, 4, 1))
    numpy.testing.assert_allclose(haar_like(a), [[-1.], [11.], [-5.]])


def check_matching(match_subregion):
    from scipy.misc import lena
    image = lena().astype('float64')[:100, :100, numpy.newaxis]
    patch = image[22:22 + 50, 44:44 + 50]
    assert match_subregion(image, patch) == (22, 44)
