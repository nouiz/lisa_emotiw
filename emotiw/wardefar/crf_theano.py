"""
Theano implementation of structured output inference in a chain-CRF
based classification model.
"""
__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley", "Yoshua Bengio"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


import theano
from theano import tensor


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.

    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).

    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.

    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + tensor.log(tensor.exp(x - xmax).sum(axis=axis))


def forward_theano(obs_potentials, chain_potentials, viterbi=False):
    """
    Given (symbolic) log-domain potentials, construct
    the graph for forward inference in a chain CRF.

    Parameters
    ----------
    obs_potentials : tensor_like (n_steps, n_classes)
        Axes correspond to time and the value of the discrete
        label variable. This is the energy assigned to a
        configuration (so higher energy = lower probability).

    chain_potentials : tensor_like (n_classes, n_classes, n_classes)
        Axes correspond to left label state, right label state,
        and the global label. Corresponds to the energy of a
        given pair of labels adjacent to one another (higher
        energy = lower probability).

    viterbi : bool, optional
        Perform MAP inference with the Viterbi algorithm rather
        than marginalizing the step-specific label variables,
        Instead, use the single most likely configuration.

    Returns
    -------
    energy : TensorVariable, 1-dimensional
        The energy assigned for a given global label. This
        can be turned into a log probability by subtracting
        logsumexp(energy).
    """
    def inner_function(obs, prior_result, chain_potentials):
        prior_result = prior_result.dimshuffle(0, 'x', 1)
        obs = obs.dimshuffle('x', 0, 'x')
        if viterbi:
            out = (-prior_result - obs - chain_potentials).max(axis=0)
        else:
            out = theano_logsumexp(-prior_result - obs - chain_potentials,
                                   axis=0)
        return out

    assert obs_potentials.ndim == 2
    assert chain_potentials.ndim == 3
    initial = (obs_potentials[0].dimshuffle(0, 'x') *
               tensor.ones_like(chain_potentials[0]))
    scanned, _ = theano.scan(fn=inner_function,
                             outputs_info=initial,
                             sequences=[obs_potentials[1:]],
                             non_sequences=chain_potentials)
    if viterbi:
        return -(scanned[-1].max(axis=0))
    else:
        return -theano_logsumexp(scanned[-1], axis=0)
