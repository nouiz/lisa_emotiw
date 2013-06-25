"""
NumPy implementation of structured output inference in a chain-CRF
based classification model.
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley", "Yoshua Bengio"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


import numpy as np


def logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.

    Parameters
    ----------
    x : array_like
        A NumPy ndarray or object that can be coerced to one.

    axis : int or None, optional
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.

    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """
    x = np.asarray(x)
    xmax = x.max(axis=axis)
    if axis is not None:
        if axis >= x.ndim or axis < -x.ndim:
            raise ValueError('invalid axis %d for %d-dimensional array' %
                             (axis, x.ndim))
        elif axis < 0:
            axis += x.ndim
        idx_tuple = ([slice(None)] * axis +
                     [None] +
                     (x.ndim - axis - 1) * [slice(None)])
    else:
        idx_tuple = Ellipsis
    return xmax + np.log(np.exp(x - xmax[idx_tuple]).sum(axis=axis))


def _forward_final_step(alpha, viterbi):
    """
    Common code for the final step of forward inference (either
    Baum-Welch or Viterbi).
    """
    if viterbi:
        # Take the max of log probability, then take the negative
        # to get units of energy.
        return -(alpha[-1].max(axis=0))
    else:
        # Take the negative after marginalizing out the state variables.
        return -logsumexp(alpha[-1], axis=0)


def forward(obs_potentials, chain_potentials, viterbi=False):
    """
    Given log-domain potentials, perform forward inference in
    a chain CRF.

    Parameters
    ----------
    obs_potentials : ndarray (n_steps, n_classes)
        Axes correspond to time and the value of the discrete
        label variable. This is the energy assigned to a
        configuration (so higher energy = lower probability).

    chain_potentials : ndarray (n_classes, n_classes, n_classes)
        Axes correspond to left label state, right label state,
        and the global label. Corresponds to the energy of a
        given pair of labels adjacent to one another (higher
        energy = lower probability).

    viterbi : bool, optional
        Perform MAP inference with the Viterbi algorithm rather
        than marginalizing the step-specific label variables,
        Instead, use the single most likely configuration. Default
        is `False`.

    Returns
    -------
    energy : ndarray, 1-dimensional
        The energy assigned for a given global label. This
        can be turned into a log probability by subtracting
        logsumexp(energy).
    """
    n_steps, n_classes = obs_potentials.shape
    # Number of time steps, state classes, actual classes
    alpha = np.empty((n_steps, n_classes, n_classes))
    alpha[...] = np.nan  # Special-cases the first step (see * below).
    # The transpose just gets the broadcasting right. We want to
    # broadcast the first row of obs_potentials across columns of alpha[0].
    # We use 0:1 to preserve ndim=2.
    alpha[0, :, :] = obs_potentials[0:1, :].T
    # Use maximum if we are doing Viterbi decoding, logaddexp otherwise.
    reducer = np.maximum if viterbi else np.logaddexp
    for t in xrange(1, n_steps):
        for glob_l in xrange(n_classes):  # Loop over "global label" state.
            for this_l in xrange(n_classes):  # Loop over state at time t.
                for prev_l in xrange(n_classes):  # Loop over states at t - 1.
                    # Assign some variables to make it easier to read.
                    a = alpha[t - 1, prev_l, glob_l]
                    c = chain_potentials[prev_l, this_l, glob_l]
                    o = obs_potentials[t, this_l]
                    # We are accumulating in this, but with logaddexp instead
                    # of just a normal addition (or max in case of Viterbi).
                    e = alpha[t, this_l, glob_l]
                    if np.isnan(e):  # * First step, so initialize.
                        alpha[t, this_l, glob_l] = -a - c - o
                    else:
                        alpha[t, this_l, glob_l] = reducer(e, -a - c - o)
    return _forward_final_step(alpha, viterbi)


def forward_vectorized(obs_potentials, chain_potentials, viterbi=False):
    """
    Given log-domain potentials, perform forward inference in
    a chain CRF.

    Notes
    -----
    See the docstring for `forward`. Identical but with the
    inner loops vectorized.
    """
    n_steps, n_classes = obs_potentials.shape
    # Number of time steps, state classes, actual classes
    alpha = np.empty((n_steps, n_classes, n_classes)) * np.nan
    # The transpose just gets the broadcasting right. We want to
    # broadcast the first row of obs_potentials across columns of alpha[0].
    # We use 0:1 to preserve ndim=2.
    alpha[0, :, :] = obs_potentials[0:1, :].T
    for t in xrange(1, n_steps):
        a = alpha[t - 1, :, np.newaxis, :]
        c = chain_potentials
        o = obs_potentials[t, np.newaxis, :, np.newaxis]
        if viterbi:
            alpha[t] = (-a - c - o).max(axis=0)
        else:
            alpha[t] = logsumexp(-a - c - o, axis=0)
    return _forward_final_step(alpha, viterbi)


