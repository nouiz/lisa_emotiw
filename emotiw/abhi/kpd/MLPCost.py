from pylearn2.costs.cost import SumOfCosts, Cost
import numpy as np
from theano import config
import theano.tensor as T
from theano import function
from theano.gof.op import get_debug_values
from theano.printing import Print

from collections import OrderedDict

from pylearn2.utils import serial
from itertools import izip
from pylearn2.utils import safe_zip


class MLPCost(Cost):
    supervised = True
    def __init__(self, cost_type='default', missing_target_value=None):
        self.__dict__.update(locals())
        del self.self
        self.use_dropout = False
    
    def setup_dropout(self, default_input_include_prob=.5, 
                        input_scales=None, input_include_probs=None, 
                        default_input_scale=2.):
        """
        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self
        
        self.use_dropout=True

    def get_gradients(self, model, X, Y=None, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()

        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """

        try:
            if Y is None:
                cost = self(model=model, X=X, **kwargs)
            else:
                cost = self(model=model, X=X, Y=Y, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither

            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'raise')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates
        
    def __call__(self, model, X, Y, ** kwargs):
        if self.use_dropout:
            Y_hat = model.dropout_fprop(X, default_input_include_prob=self.default_input_include_prob,
                    input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                    input_scales=self.input_scales
                    )
        else:
            Y_hat = model.fprop(X)
        
        if self.missing_target_value is not None:
            assert (self.cost_type == 'default')
            costMatrix = model.layers[-1].cost_matrix(Y, Y_hat)
            costMatrix *= T.neq(Y, -1)  # This sets to zero all elements where Y == -1
            cost = costMatrix.sum(axis=2).sum()
            #cost = T.cast(cost, 'float32')
            #cost = model.cost_from_cost_matrix(costMatrix)
        else:
            if self.cost_type == 'default':
                cost = model.cost(Y, Y_hat)
            elif self.cost_type == 'nll':
                cost = (-Y * T.log(Y_hat)).sum(axis=1).mean()
            elif self.cost_type == 'crossentropy':
                cost = (-Y * T.log(Y_hat) - (1 - Y) \
                    * T.log(1 - Y_hat)).sum(axis=1).mean()
            else:
                raise NotImplementedError()
        return cost
        
    def get_test_cost(self, model, X, Y):
        use_dropout = self.use_dropout
        self.use_dropout = False
        cost = self.__call__(model, X, Y)
        self.use_dropout = use_dropout
        return cost
