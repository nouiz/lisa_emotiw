"""
Tests for the NumPy implementation of the CRF classifier.
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


import numpy as np
from .crf_numpy import forward, forward_vectorized, logsumexp


def test_logsumexp():
    x = -50000. - np.arange(1, 4) / 10.
    np.testing.assert_allclose(logsumexp(x), -49999.098057151772)
    y = 50000. + np.arange(1, 4) / 10.
    np.testing.assert_allclose(logsumexp(y), 50001.301942848229)
    z = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    np.testing.assert_allclose(logsumexp(z, axis=0), [-49999.098057151772,
                                                      50001.301942848229])
    np.testing.assert_allclose(logsumexp(z.T, axis=1), [-49999.098057151772,
                                                        50001.301942848229])
    np.testing.assert_allclose(logsumexp(z, axis=-2), [-49999.098057151772,
                                                       50001.301942848229])
    np.testing.assert_allclose(logsumexp(z.T, axis=-1), [-49999.098057151772,
                                                         50001.301942848229])


def test_forward():
    global_0 = np.array([[4., 6.], [5., 7.]])
    global_1 = np.array([[8., 10.], [9., 11.]])
    chain = np.concatenate((global_0[..., np.newaxis],
                            global_1[..., np.newaxis]), axis=2)
    obs = np.array([[0, 1.], [2., 3.]])
    # This is in units of log probability, i.e. negative energy.
    expected = -np.array([[6., 10.], [9., 13.]]) + np.log1p(np.exp(-2))
    # And this negative brings it to units of energy.
    expected = -logsumexp(expected, axis=0)
    actual = forward(obs, chain)
    np.testing.assert_allclose(actual, expected)
    viterbi = forward(obs, chain, viterbi=True)
    np.testing.assert_allclose(viterbi, [6., 10.])


def test_vectorized():
    rng = np.random.RandomState([2013, 6, 1])
    for i in range(20):
        num_labels = rng.random_integers(2, 10)
        num_timesteps = rng.random_integers(2, 10)
        obs = rng.uniform(size=(num_timesteps, num_labels))
        chain = rng.uniform(size=(num_labels, num_labels, num_labels))
        np.testing.assert_allclose(forward_vectorized(obs, chain),
                                   forward(obs, chain))
        np.testing.assert_allclose(forward_vectorized(obs, chain,
                                                      viterbi=True),
                                   forward(obs, chain, viterbi=True))
