from theano import tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace

class MulticlassMargin(Cost):
    """
        Same principle as SVM
    """

    def expr(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
#        print "Y",
#        print Y.dtype

#        Y = np.cast[theano.config.floatX](Y)

        Y_hat = model.fprop(X)
#        print "Y_hat",
#        print Y_hat.dtype

        Y_others = Y_hat + Y * -1000000.
#        print "Y_others",
#        print Y_others.dtype
        cost = 1 + T.max(Y_others,axis=1) - (Y*Y_hat).sum(1)
#        print "cost",
#        print cost.dtype

#        from theano import function
#        import numpy as np

#        f = function([X,Y],cost)
#        x = np.cast[theano.config.floatX](np.zeros((20,49)))
#        y = np.cast[theano.config.floatX](np.zeros((20,7)))
#        y[:,0] = 1.0
#        print f(x,y)
#        print f(x,y).shape
#        import sys
#        sys.exit(0)

        return (cost * (cost > 0)).mean()

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)


