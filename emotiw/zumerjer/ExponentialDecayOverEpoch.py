from pylearn2.train_extensions import TrainExtension
import numpy as np
from theano import config
import theano.tensor as T
from theano import function
from theano.gof.op import get_debug_values
from theano.printing import Print



class ExponentialDecayOverEpoch(TrainExtension):
    """
    This is a callback for the SGD algorithm rather than the Train obj.
    This anneals the lr by dividing by decay_factor after each
    epoch. It will not shrink the learning rate beyond 
    min_lr_scale*learning_rate.
    """
    def __init__(self, decay_factor, min_lr_scale):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(min_lr_scale, str):
            min_lr_scale = float(min_lr_scale)
        assert isinstance(decay_factor, float)
        assert isinstance(min_lr_scale, float)
        self.__dict__.update(locals())
        del self.self
        self._count = 0
        
    def on_monitor(self, model, dataset, algorithm):
        if self._count == 0:
            self._cur_lr = algorithm.learning_rate.get_value()
            self.min_lr = self._cur_lr*self.min_lr_scale
        self._count += 1
        self._cur_lr = max(self._cur_lr * self.decay_factor, self.min_lr)
        algorithm.learning_rate.set_value(np.cast[config.floatX](self._cur_lr))

    def __call__(self, algorithm):
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
        self._count += 1
        cur_lr = self._base_lr / (self.decay_factor ** self._count)
        new_lr = max(cur_lr, self.min_lr)
        new_lr = np.cast[config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)   
