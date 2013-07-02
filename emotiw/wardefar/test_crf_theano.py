"""
Tests for the Theano CRF classifier implementation.
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


import numpy as np
import theano
from .crf_theano import theano_logsumexp, forward_theano
from .crf_numpy import forward_vectorized


def test_theano_logsumexp():
    x_ = theano.tensor.vector()
    f = theano.function([x_], theano_logsumexp(x_))
    x = -50000. - np.arange(1, 4) / 10.
    np.testing.assert_allclose(f(x), -49999.098057151772)
    y = 50000. + np.arange(1, 4) / 10.
    np.testing.assert_allclose(f(y), 50001.301942848229)
    z = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    y_ = theano.tensor.matrix()
    g = theano.function([y_], theano_logsumexp(y_, axis=0))
    np.testing.assert_allclose(g(z), [-49999.098057151772,
                                      50001.301942848229])
    g = theano.function([y_], theano_logsumexp(y_, axis=1))
    np.testing.assert_allclose(g(z.T), [-49999.098057151772,
                                        50001.301942848229])
    # Negative indices make Theano barf at the moment.
    # (Theano ticket gh-1430)
    # g = theano.function([y_], theano_logsumexp(y_, axis=-2))
    # np.testing.assert_allclose(g(z), [-49999.098057151772,
    #                                   50001.301942848229])
    # np.testing.assert_allclose(g(z.T), [-49999.098057151772,
    #                                     50001.301942848229])


def test_forward_theano():
    rng = np.random.RandomState([2013, 6, 1])
    o = theano.tensor.matrix()
    c = theano.tensor.tensor3()
    f = theano.function([o, c], forward_theano(o, c))
    g = theano.function([o, c], forward_theano(o, c, viterbi=True))
    for i in range(20):
        num_labels = rng.random_integers(2, 10)
        num_timesteps = rng.random_integers(2, 10)
        obs = rng.uniform(size=(num_timesteps, num_labels))
        chain = rng.uniform(size=(num_labels, num_labels, num_labels))
        np.testing.assert_allclose(f(obs, chain),
                                   forward_vectorized(obs, chain))
        np.testing.assert_allclose(g(obs, chain),
                                   forward_vectorized(obs, chain,
                                                      viterbi=True))
