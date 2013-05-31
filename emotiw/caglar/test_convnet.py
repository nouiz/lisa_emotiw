from pylearn2.train import Train
from pylearn2.models.mlp import MLP, ConvRectifiedLinear
#from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.models.maxout import Maxout
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
from pylearn2.space import Conv2DSpace

from pylearn2.models.mlp import Sigmoid
from face_bbox import FaceBBox
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

MAX_EPOCHS = 20

def get_dataset():
    which_set = "train"
    path = "/data/lisatmp/data/faces_bbox/test_face_lvl1.h5"
    start = 0
    stop = 2100
    size_of_receptive_field = [88, 88]
    stride = 8
    use_output_map = True
    dataset = FaceBBox(which_set=which_set,
                        start=start,
                        stop=stop,
                        use_output_map=use_output_map,
                        stride=stride,
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

    return Train(model = model,
            algorithm = train_algo,
            extensions = extensions,
            dataset = trainset)

def test_convnet():
    layers = []
    dataset = get_dataset()
    input_space = Conv2DSpace(
            shape=[256, 256],
            num_channels=1)

    conv_layer = ConvRectifiedLinear(
                output_channels=12,
                irange=.005,
                layer_name="h0",
                kernel_shape=[88, 88],
                kernel_stride=[8, 8],
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
    sigmoid_layer = Sigmoid(layer_name="y",
                    dim=484,
                    monitor_style="detection",
                    irange=.005)

    layers.append(sigmoid_layer)
    model = MLP(batch_size=100,
            layers = layers,
            input_space = input_space)

    trainer = get_layer_trainer_sgd(model, dataset)
    trainer.main_loop()

if __name__=="__main__":
    test_convnet()
