"""
Test that a smaller version of convolutional_network.ipynb works.

If this file has to be edited, convolutional_network.ipynb has to be updated
in the same way.

The differences (needed for speed) are:
    * output_channels: 4 instead of 64
    * train.stop: 500 instead of 50000
    * valid.stop: 50100 instead of 60000
    * test.start: 0 instead of non-specified
    * test.stop: 100 instead of non-specified
    * termination_criterion.prop_decrease: 0.50 instead of 0.0

This should make the test run in about one minute.
"""
from nose.plugins.skip import SkipTest

from pylearn2.datasets.exc import NoDataPathError
from pylearn2.testing import no_debug_mode


@no_debug_mode
def train_convolutional_network():
    train = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:emotiw.caglar.face_bbox.FaceBBox {
        which_set: 'train',
        start: 0,
        stop: 1000,
        area_ratio: 0.90,
        img_shape: [128, 128],
        size_of_receptive_field: [98, 98],
        stride: 1,
        path: "/data/lisatmp/data/faces_bbox/test_face_lvl2.h5",
        use_output_map: True
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 100,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [128, 128],
            num_channels: 1
        },
        layers: [
                    !obj:pylearn2.models.mlp.ConvSigmoid {
                        layer_name: 'h0',
                        output_channels: 128,
                        irange: .05,
                        kernel_shape: [98, 98],
                        max_kernel_norm: 1.9365
                    }
                ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .08,
        init_momentum: .6,
        train_iteration_mode: 'batchwise_shuffled_equential',
        monitoring_dataset:
            {
                'train'  : *train
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X'
            }, !obj:pylearn2.costs.mlp.WeightDecay {
                coeffs: [ .00005 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "train_h0_kl",
            prop_decrease: 0.50,
            N: 10
        }
    },
    extensions:
        [ !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'train_h0_kl',
             save_path: "sigmoid_convnet_best.pkl"
           },
           !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ]
}
"""

    from pylearn2.config import yaml_parse
    train = yaml_parse.load(train)
    train.main_loop()


def test_convolutional_network():
    try:
        train_convolutional_network()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")


if __name__ == '__main__':
    test_convolutional_network()
