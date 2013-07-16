from pylearn2.train import Train
from emotiw.common.datasets.faces.afew2_facetubes import AFEW2FaceTubes
from pylearn2.model.mlp import MLP, Softmax
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

save_path= "last.pkl"
save_best_path = 'best.pkl'
save_freq= 10



train_ds = AFEW2FaceTubes(
        which_set= 'train',
        one_hot= 1,
        preproc= ['smooth'],
        size= [48, 48])

print 'Train Dataset Loaded'

val_ds = AFEW2FaceTubes(
         which_set= 'val',
         one_hot= 1,
         preproc= ['smooth'],
         size= [48, 48])

print 'Val Dataset Loaded'
last_ndim = 240
n_classes = 7


algorithm= SGD(
        batch_size= 1,
        learning_rate= 0.000100,
        init_momentum= .5,
        monitoring_dataset= {'valid' : val_ds},
        cost= Dropout(input_include_probs= { 'h0' : .8 },
                      input_scales= { 'h0': 1. }),
        termination_criterion= MonitorBased(
            channel_name= "valid_y_misclass",
            prop_decrease= 0.,
            N=100),
        #termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {max_epochs: 1},
        update_callbacks = ExponentialDecay(
            decay_factor= 1.00004,
            min_lr= .000001)
    )

extensions = [MonitorBasedSaveBest(
             channel_name= 'valid_y_misclass',
             save_path= save_best_path),
             MomentumAdjustor(
            start= 1,
            saturate= 250,
            final_momentum= .7)]
    
model = MLP(input_space = Conv2DSpace (
            shape= [48, 48],
            num_channels= 3,
            axes= ['b', 't', 0, 1, 'c']
            ),
             layers= [MaxoutConvC01B(
                     layer_name= 'h0',
                     pad= 0,
                     num_channels= 64,
                     num_pieces= 2,
                     kernel_shape= [8, 8],
                     pool_shape= [4, 4],
                     pool_stride= [2, 2],
                     irange= .005,
                     max_kernel_norm= .9,
                     W_lr_scale= 0.5,
                     b_lr_scale= 0.5
                 ),  MaxoutConvC01B (
                     layer_name= 'h1',
                     pad= 0,
                     num_channels= 64,
                     num_pieces= 2,
                     kernel_shape= [8, 8],
                     pool_shape= [3, 3],
                     pool_stride= [2, 2],
                     irange= .005,
                     max_kernel_norm= .9,
                     W_lr_scale= 0.5,
                     b_lr_scale= 0.5
                 ),  Maxout(layer_name= 'h2',
                     num_units= last_ndim,
                     num_pieces= 2,
                     irange= .005
                 ),  Softmax(
                     max_col_norm= 1.9365,
                     layer_name= 'y',
                     n_classes= n_classes,
                     sparse_init= 23
                     )
                     ])

train = Train(
    dataset= train_ds, 
    model=model,
    algorithm=algorithm,
    extensions= extensions,
    save_path= save_path,
    save_freq=save_freq
)
