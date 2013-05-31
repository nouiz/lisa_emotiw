import numpy


class MultiScaleDistinctMLP(object):
    """
        Implementation of MultiScale Distinct Convnet
        We assume that we are trying N different convolutional neural networks
        on different scales without any parameter sharing.
    """
    def __init__(self, trainers, models, pyramids, nlevels, seed=None):
        self.models = models
        self.trainers = trainers
        self.pyramids = pyramids
        self.algorithms = [ trainers[i].algorithm for i in xrange(len(trainers))]

        assert len(trainers) == nlevels
        assert len(pyramids) == nlevels
        assert len(models) == nlevels

        self.nlevels = nlevels

        self._setup_trainers()

        if seed is None:
            seed = [2013, 5, 28]
        self.seed = seed

    def _setup_trainers(self):
        for i in xrange(self.nlevels):
            self.algorithms[i].setup(self.models[i], self.pyramids[i])

    def frop(self, X, models_no):
        if models_no >= self.nlevels:
            raise ValueError("Number of levels should be smaller than the number of models.")
        below = X
        outputs = []
        model = self.models[models_no]
        for layer in model.layers:
            above = layer.fprop(below)
            below = above
            outputs.append(above)
        assert len(outputs) > 0
        return outputs

    def train(self, models_no):
        self.trainers[models_no].main_loop()

class MultiScaleSharedMLP(object):
    """
        Implementation of Convnet with shared convolutional layers.
        This implementation has a layer that has shared first N layers are shared
        across different models.
        n_shared_layer: Number of convolution layers.
    """
    def __init__(self, n_shared_layer, trainers, models, pyramids, nlevels, seed=None):
        self.models = models
        self.trainers = trainers
        self.pyramids = pyramids
        self.n_shared_layer = n_shared_layer
        self.algorithms = [ trainers[i].algorithm for i in xrange(len(trainers))]

        assert len(trainers) == nlevels
        assert len(pyramids) == nlevels
        assert len(models) == nlevels

        self._check_param_sharing()
        self._share_params()
        self._setup_trainers()

        self.nlevels = nlevels
        if seed is None:
            seed = [2013, 5, 28]
        self.seed = seed

    def _setup_trainers(self):
        for i in xrange(self.nlevels):
            self.algorithms[i].setup(self.models[i], self.pyramids[i])

    def _check_param_sharing(self):
        for model in self.models:
            for i in xrange(0, self.n_shared_layer):
                if not (model.layers[i].can_alter_transformer and
                        model.layers[i].can_alter_filters):
                    raise ValueError("models is not supporting the parameter sharing.")

    def _share_params(self):
        first_model = self.models[0]
        filters = first_model.get_weights()
        biases = first_model.get_biases()

        for i in xrange(1, self.nlevels):
            for j in xrange(0, self.n_shared_layer):
                self.models[i].layers[j].set_shared_filters(filters)
                self.models[i].layers[j].set_shared_biases(biases)

    def frop(self, X, models_no):
        if models_no >= self.nlevels:
            raise ValueError("Number of levels should be smaller than the number of models.")
        below = X
        outputs = []
        model = self.models[models_no]
        for layer in model.layers:
            above = layer.fprop(below)
            below = above
            outputs.append(above)
        assert len(outputs) > 0
        return outputs

    def train(self, X, targets, models_no):
        self.trainers[models_no].main_loop()

