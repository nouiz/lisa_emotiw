
import math
import sys
import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.models.mlp import *


class ConvSigmoid(Layer):
    """
    Convolutional layer with sigmoid nonlinearity.
    """
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 irange = None,
                 border_mode = 'valid',
                 monitor_style = "detection",
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_kernel_norm = None,
                 pool_type = None,
                 pool_shape = None,
                 pool_stride = None,
                 can_alter_transformer = True,
                 can_alter_biases = True,
                 is_final_layer = True,
                 detector_normalization = None,
                 output_normalization = None,
                 kernel_stride=(1, 1)):
        """
                 output_channels: The number of output channels the layer should have.
                 kernel_shape: The shape of the convolution kernel.
                 pool_shape: The shape of the spatial max pooling. A two-tuple of ints.
                 pool_stride: The stride of the spatial max pooling. Also must be square.
                 layer_name: A name for this layer that will be prepended to
                 monitoring channels related to this layer.
                 irange: if specified, initializes each weight randomly in
                 U(-irange, irange)
                 border_mode:A string indicating the size of the output:
                    full - The output is the full discrete linear convolution of the inputs.
                    valid - The output consists only of those elements that do not rely
                    on the zero-padding.(Default)
                 include_prob: probability of including a weight element in the set
            of weights initialized to U(-irange, irange). If not included
            it is initialized to 0.
                 init_bias: All biases are initialized to this number
                 W_lr_scale:The learning rate on the weights for this layer is
                 multiplied by this scaling factor
                 b_lr_scale: The learning rate on the biases for this layer is
                 multiplied by this scaling factor
                 max_kernel_norm: If specifed, each kernel is constrained to have at most this
                 norm.

                 can_alter_transformer: Flag that determines if the transformer is changeable.
                 This flag can be useful for sharing filters across different layers with
                 set_shared_filters function.

                 can_alter_biases: Flag that determines if the biases are changeable or
                 not. This flag can be useful for bias sharing across different layers
                 with set_shared_biases function.

                 pool_type: The type of the pooling operation performed the the convolution.
                 Default pooling type is max-pooling.
                 detector_normalization, output_normalization:
                      if specified, should be a callable object. the state of the network is
                      optionally
         replaced with normalization(state) at each of the 3 points in processing:
                          detector: the maxout units can be normalized prior to the spatial
                          pooling
                          output: the output of the layer, after sptial pooling, can be normalized
                          as well
                 kernel_stride: The stride of the convolution kernel. A two-tuple of ints.
        """

        if (irange is None):
            raise AssertionError("You should specify either irange when calling the constructor of ConvRectifiedLinear.")

        self.__dict__.update(locals())
        assert monitor_style in ['classification', 'detection']
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space
        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0]) / self.kernel_stride[0] + 1,
                (self.input_space.shape[1] - self.kernel_shape[1]) / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] +  self.kernel_shape[0]) / self.kernel_stride[0] - 1,
                    (self.input_space.shape[1] + self.kernel_shape[1]) / self.kernel_stride_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('b', 'c', 0, 1))

        if not (self.can_alter_transformer and hasattr(self, "transformer")
                and self.transformer is not None):

            if self.irange is not None:
                self.transformer = conv2d.make_random_conv2D(
                        irange = self.irange,
                        input_space = self.input_space,
                        output_space = self.detector_space,
                        kernel_shape = self.kernel_shape,
                        batch_size = self.mlp.batch_size,
                        subsample = self.kernel_stride,
                        border_mode = self.border_mode,
                        rng = rng)
        else:
            filters_shape = self.transformer._filters.get_value().shape
            if (self.input_space.filters_shape[-2:-1] != self.kernel_shape or
                    self.input_space.filters_shape != filters_shape):
                raise ValueError("The filters and input space don't have compatible input space.")
            if self.input_space.num_channels != filters_shape[1]:
                raise ValueError("The filters and input space don't have compatible number of channels.")

        W, = self.transformer.get_params()
        W.name = 'W'

        if not (self.can_alter_biases and hasattr(self, "b")
                and self.b is not None):
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
            self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape


        dummy_batch_size = self.mlp.batch_size
        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector = sharedX(self.detector_space.get_origin_batch(dummy_batch_size))

        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[2], dummy_p.shape[3]],
                    num_channels = self.output_channels, axes = ('b', 'c', 0, 1) )
        else:
            dummy_detector = dummy_detector.eval()
            self.output_space = Conv2DSpace(shape=[dummy_detector.shape[2], dummy_detector.shape[3]],
                    num_channels = self.output_channels, axes = ('b', 'c', 0, 1) )

        print 'Output space: ', self.output_space.shape


    def censor_updates(self, updates):
        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1,2,3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * abs(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp,rows,cols,inp))

    def get_detection_channels_from_state(self, state, target):

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)
        rval['01_loss'] = wrong_bit.mean()
        rval['kl'] = self.cost(Y_hat=state, Y=target)
        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat).sum()
        fp = ((1-y) * y_hat).sum()
        precision = tp / T.maximum(1., tp + fp)
        recall = tp / T.maximum(1., y.sum())
        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = 2. * precision * recall / T.maximum(1, precision + recall)

        tp = (y * y_hat).sum(axis=[0, 1])
        fp = ((1-y) * y_hat).sum(axis=[0, 1])
        precision = tp / T.maximum(1., tp + fp)

        rval['per_output_precision.max'] = precision.max()
        rval['per_output_precision.mean'] = precision.mean()
        rval['per_output_precision.min'] = precision.min()

        recall = tp / T.maximum(1., y.sum(axis=[0, 1]))

        rval['per_output_recall.max'] = recall.max()
        rval['per_output_recall.mean'] = recall.mean()
        rval['per_output_recall.min'] = recall.min()

        f1 = 2. * precision * recall / T.maximum(1, precision + recall)

        rval['per_output_f1.max'] = f1.max()
        rval['per_output_f1.mean'] = f1.mean()
        rval['per_output_f1.min'] = f1.min()
        return rval

    def get_monitoring_channels_from_state(self, state, target=None):

        rval = super(ConvSigmoid, self).get_monitoring_channels_from_state(state, target)

        if target is not None:
            if self.monitor_style == 'detection':
                rval.update(self.get_detection_channels_from_state(state, target))
            else:
                assert self.monitor_style == 'classification'
                # Threshold Y_hat at 0.5.
                prediction = T.gt(state, 0.5)
                # If even one feature is wrong for a given training example,
                # it's considered incorrect, so we max over columns.
                incorrect = T.neq(target, prediction).max(axis=1)
                rval['misclass'] = T.cast(incorrect, config.floatX).mean()
        return rval

    def get_monitoring_channels(self):
        W ,= self.transformer.get_params()
        assert W.ndim == 4
        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=(1,2,3)))
        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            ])

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = T.nnet.sigmoid(z)

        self.detector_space.validate(d)

        if not hasattr(self, 'detector_normalization'):
            self.detector_normalization = None

        if self.detector_normalization:
            d = self.detector_normalization(d)
        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                p = max_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)
        else:
            p = d

        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    def cost(self, Y, Y_hat):
        """
        Note : Method copied from the Sigmoid class

        mean across units, mean across batch of KL divergence
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        KL(P || Q) = p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        """

        ave_total = kl(Y, Y_hat)
        ave = ave_total.mean()
        return ave


def kl(Y, Y_hat):
    """
    Returns a batch (vector) of
    mean across units of KL divergence for each example
    KL(P || Q) where P is defined by Y and Q is defined by Y_hat
    Currently Y must be purely binary. If it's not, you'll still
    get the right gradient, but the value in the monitoring channel
    will be wrong.
    Y_hat must be generated by fprop, i.e., it must be a symbolic
    sigmoid.

    p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
    For binary p, some terms drop out:
    - p log q - (1-p) log (1-q)
    - p log sigmoid(z) - (1-p) log sigmoid(-z)
    p softplus(-z) + (1-p) softplus(z)
    """
    # Pull out the argument to the sigmoid
    assert hasattr(Y_hat, 'owner')
    owner = Y_hat.owner
    assert owner is not None
    op = owner.op

    if not hasattr(op, 'scalar_op'):
        owner = Y_hat.owner.inputs[0].owner
        assert owner is not None
        op = owner.op

    if not hasattr(op, 'scalar_op'):
        raise ValueError("Expected Y_hat to be generated by an Elemwise op, got "+str(op)+" of type "+str(type(op)))

    assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
    z ,= owner.inputs

    term_1 = Y * T.nnet.softplus(-z)
    term_2 = (1 - Y) * T.nnet.softplus(z)

    total = term_1 + term_2

    ave_ch = total.mean(axis=1)
    assert ave_ch.ndim == 3

    ave_x = ave_ch.mean(axis=1)
    assert ave_x.ndim == 2

    ave_y = ave_x.mean(axis=1)
    assert ave_y.ndim == 1

    return ave_y
