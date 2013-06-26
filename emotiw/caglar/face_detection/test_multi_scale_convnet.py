from pylearn2.train import Train
from pylearn2.models.mlp import MLP, ConvRectifiedLinear
from pylearn2.models.maxout import Maxout
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
from pylearn2.space import Conv2DSpace

from pylearn2.models.mlp import Sigmoid
from face_bbox import FaceBBox
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

from multi_scale_convnet import *

MAX_EPOCHS = 20

def get_dataset(which_set = "train",
        path = "/data/lisatmp/data/faces_bbox/test_face_lvl1.h5",
        start = 0,
        stop = 2100,
        img_shape = [256, 256],
        size_of_receptive_field = [88, 88],
        stride = 8,
        use_output_map = True):

    dataset = FaceBBox(which_set=which_set,
                        start=start,
                        stop=stop,
                        use_output_map=use_output_map,
                        stride=stride,
                        img_shape=img_shape,
                        size_of_receptive_field=size_of_receptive_field,
                        path=path)

    return dataset

def get_layer_trainer_sgd(model, trainset):
    drop_cost = Dropout(
        input_include_probs={ 'h0': .4},
        input_scales={'h0': 1.})

    # configs on sgd
    train_algo = SGD(
              train_iteration_mode='batchwise_shuffled_equential',
              learning_rate = 0.2,
              cost = drop_cost,
              monitoring_dataset = trainset,
              termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS),
              update_callbacks = None)

    extensions = [MonitorBasedSaveBest(channel_name="y_kl",
                save_path="./convnet_test_best.pkl")]

    return Train(algorithm = train_algo,
                dataset = trainset,
                model = model,
                extensions = extensions)

def get_convnet(img_shape=[256, 256], output_channels=16, kernel_shape=[88, 88], kernel_stride=[8,
    8]):
    layers = []
    dataset = get_dataset()
    input_space = Conv2DSpace(
            shape=img_shape,
            num_channels=1)

    conv_layer = ConvRectifiedLinear(
                output_channels=output_channels,
                irange=.005,
                layer_name="h0",
                kernel_shape=kernel_shape,
                kernel_stride=kernel_stride,
                pool_shape=[1, 1],
                pool_stride=[1, 1],
                max_kernel_norm=1.932)

    layers.append(conv_layer)

    maxout_layer = Maxout(layer_name="h1",
                    irange=.005,
                    num_units=600,
                    num_pieces=4,
                    max_col_norm=1.932)

    layers.append(maxout_layer)
    conv_out_dim = ((img_shape[0] - kernel_shape[0])/kernel_stride[0] + 1)**2
    sigmoid_layer = Sigmoid(layer_name="y",
                    dim=conv_out_dim,
                    monitor_style="detection",
                    irange=.005)

    layers.append(sigmoid_layer)

    model = MLP(batch_size=100,
            layers = layers,
            input_space = input_space)
    return model

if __name__=="__main__":
    trainers = []
    models = []
    datasets = []

    dataset1 = get_dataset()

    dataset2 = get_dataset(path = "/data/lisatmp/data/faces_bbox/test_face_lvl2.h5",
        start = 0,
        stop = 2100,
        img_shape = [128, 128],
        size_of_receptive_field = [48, 48],
        stride = 4)

    datasets.append(dataset1)
    datasets.append(dataset2)

    model1 = get_convnet()
    model2 = get_convnet(img_shape=[128, 128], output_channels=16, kernel_shape=[48, 48],
            kernel_stride=[4, 4])

    models.append(model1)
    models.append(model2)


    trainers.append(get_layer_trainer_sgd(model1, dataset1))
    trainers.append(get_layer_trainer_sgd(model2, dataset2))

    mcovnet = MultiScaleDistinctConvnet(trainers, models, datasets, 2)
    mcovnet.train()

