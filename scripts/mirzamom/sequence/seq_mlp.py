import numpy as np

import theano
from theano import config
from theano import tensor
from theano.printing import Print
from theano.compat.python2x import OrderedDict

from pylearn2.costs.cost import Cost
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.models import Model
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

from emotiw.common.datasets.faces.afew2_facetubes import AFEW2FaceTubes
from emotiw.common.datasets.faces.facetubes import FaceTubeSpace
from emotiw.wardefar.crf_theano import forward_theano as crf


class FrameMax(Model):
    """Frame based classifier, then elementwise max on top of representaions,
    and final classifier on top"""
    def __init__(self, mlp, final_layer, n_classes = None, input_source='features', input_space=None):
        """
        Parameters
        ----------
        mlp: Pylearn2 MLP class
            The frame based classifier

        final_layer: Pylearn2 MLP class
            Sequence based classifier
        """

        if n_classes is None:
            if hasattr(mlp.layers[-1], 'dim'):
                self.n_classes = mlp.layers[-1].dim
            elif hasattr(mlp.layers[-1], 'n_classes'):
                self.n_classes = mlp.layers[-1].n_classes
            else:
                raise ValueError("n_classes was not provided and couldn't be infered from the mlp's last layer")
        else:
            self.n_classes = n_classes

        self.mlp = mlp
        self.final_layer = final_layer
        self.input_source = input_source
        assert isinstance(input_space, FaceTubeSpace)
        self.input_space = input_space
        self.input_size = (input_space.shape[0]
                           * input_space.shape[1]
                           * input_space.num_channels)
        self.output_space = VectorSpace(dim=7)
        #self.final_layer.input_space = self.mlp.layers[-1].get_output_space()

        self.W = theano.shared(np.zeros((n_classes, n_classes, n_classes),
                                        dtype=config.floatX))
        self.W.name = 'crf_w'
        self.name = 'crf'

    def fprop(self, inputs):

        # format inputs
        inputs = self.input_space.format_as(inputs, self.mlp.input_space)
        rval = self.mlp.fprop(inputs)
        rval = tensor.max(rval, axis=0)
        rval = rval.dimshuffle('x', 0)
        rval = self.final_layer.fprop(rval)
        #if self.mlp.output_space != self.detector_space:
            #rval = self.mlp.output_space.formt_as(self.detector_space)

        #import ipdb
        #ipdb.set_trace()
        #rval = crf(rval, self.W)

        return rval

    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                    input_include_probs=None, default_input_scale=2.,
                    input_scales=None, per_example=True):

        state_below = self.input_space.format_as(state_below, self.mlp.input_space)
        rval = self.mlp.dropout_fprop(state_below, default_input_include_prob,
                    input_include_probs, default_input_scale,
                    input_scales, per_example)
        rval = tensor.max(rval, axis=0)
        rval = rval.dimshuffle('x', 0)
        rval = self.final_layer.dropout_fprop(rval, default_input_include_prob,
                    input_include_probs, default_input_scale,
                    input_scales, per_example)

        return rval

    def get_params(self):
        #return self.mlp.get_params() + [self.W]
        return self.mlp.get_params() + self.final_layer.get_params()

    def get_input_source(self):
        return self.input_source

    def get_input_space(self):
        return self.input_space

    def get_monitoring_data_specs(self):
        space = CompositeSpace((self.get_input_space(),
                                VectorSpace(dim=7)))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)


    def get_monitoring_channels(self, data):

        X, Y = data
        X = self.input_space.format_as(X, self.mlp.input_space)
        X = self.mlp.fprop(X)
        X = tensor.max(X, axis=0)
        X = X.dimshuffle('x', 0)
        return self.final_layer.get_monitoring_channels((X, Y))

    # TODO make the monitos work for main mlp
    #def get_monitoring_channels(self, data):

        #X, Y = data
        #X = self.input_space.format_as(X, self.mlp.input_space)

        ##first mlp monitors
        #ch = self.mlp.get_monitoring_channels((X, Y))

        ## get final mlp monitors
        #X = self.mlp.fprop(X)
        #X = tensor.max(X, axis=0)
        #X = X.dimshuffle('x', 0)
        #second_ch = self.final_layer.get_monitoring_channels((X, Y))

        #for key in second_ch:
            #ch[key] = second_ch[key]

        #return ch

    def cost(self, Y, Y_hat):
        return self.final_layer.cost(Y, Y_hat)


