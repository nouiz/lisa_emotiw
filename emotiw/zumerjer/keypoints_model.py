from Multisoftmax import MultiSoftmax
from MLPCost import MLPCost
from KeypointSGD import KeypointSGD
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.sgd import MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased
from pylearn2.models.maxout import Maxout
from pylearn2.training_algorithms.sgd import LinearDecayOverEpoch
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import ConvRectifiedLinear, MLP
from KeypointADADELTA import KeypointADADELTA

#params = __import__('experiment_' + sys.argv[1] + '_params')

from comboDS import ComboDatasetPyTable

# The number of features in the Y vector
numberOfKeyPoints = 98*2


def main():

    #creating layers
        #2 convolutional rectified layers, border mode valid
    batch_size = 64
    lr = 1.0 #0.1/4
    finMomentum = 0.9
    maxout_units = 2000
    num_pcs = 4
    lay1_reg = lay2_reg = maxout_reg = None
    #save_path = './models/no_maxout/titan_lr_0.1_btch_64_momFinal_0.9_maxout_2000_4.joblib'
    #best_path = '/models/no_maxout/titan_bart10_gpu2_best.joblib'
    #save_path = './models/'+params.host+'_'+params.device+'_'+sys.argv[1]+'.joblib'
    #best_path = './models/'+params.host+'_'+params.device+'_'+sys.argv[1]+'best.joblib'
    save_path = '/Tmp/zumerjer/bart10_adadelta.joblib'
    best_path = '/Tmp/zumerjer/bart10_adadelta_best.joblib'

    #numBatches = 400000/batch_size

    '''
    print 'Applying preprocessing'
    ddmTrain = EmotiwKeypoints(start=0, stop =40000)
    ddmValid = EmotiwKeypoints(start=40000, stop = 44000)
    ddmTest = EmotiwKeypoints(start=44000)

    stndrdz = preprocessing.Standardize()
    stndrdz.applyLazily(ddmTrain, can_fit=True, name = 'train')
    stndrdz.applyLazily(ddmValid, can_fit=False, name = 'val')
    stndrdz.applyLazily(ddmTest, can_fit=False, name = 'test')

    GCN = preprocessing.GlobalContrastNormalization(batch_size = 1000)
    GCN.apply(ddmTrain, can_fit =True, name = 'train')
    GCN.apply(ddmValid, can_fit =False, name = 'val')
    GCN.apply(ddmTest, can_fit = False, name = 'test')
    return
    '''

    ddmTrain = ComboDatasetPyTable('/Tmp/zumerjer/all_', which_set='train')
    ddmValid = ComboDatasetPyTable('/Tmp/zumerjer/all_', which_set='valid')
    ddmSmallTrain = ComboDatasetPyTable('/Tmp/zumerjer/all_', which_set='small_train')

    layer1 = ConvRectifiedLinear(layer_name = 'convRect1',
                     output_channels = 64,
                     irange = .05,
                     kernel_shape = [5, 5],
                     pool_shape = [4, 4],
                     pool_stride = [2, 2],
                     W_lr_scale = 0.1,
                     max_kernel_norm = lay1_reg)
    layer2 = ConvRectifiedLinear(layer_name = 'convRect2',
                     output_channels = 128,
                     irange = .05,
                     kernel_shape = [5, 5],
                     pool_shape = [3, 3],
                     pool_stride = [2, 2],
                     W_lr_scale = 0.1,
                     max_kernel_norm = lay2_reg)

        # Rectified linear units
    #layer3 = RectifiedLinear(dim = 3000,
    #                         sparse_init = 15,
    #                 layer_name = 'RectLin3')

    #Maxout layer
    maxout = Maxout(layer_name= 'maxout',
                    irange= .005,
                    num_units= maxout_units,
                    num_pieces= num_pcs,
                    W_lr_scale = 0.1,
                    max_col_norm= maxout_reg)

    #multisoftmax
    n_groups = 196
    n_classes = 96
    layer_name = 'multisoftmax'
    layerMS = MultiSoftmax(n_groups=n_groups,irange = 0.05, n_classes=n_classes, layer_name= layer_name)

    #setting up MLP
    MLPerc = MLP(batch_size = batch_size,
                 input_space = Conv2DSpace(shape = [96, 96],
                 num_channels = 3, axes=('b', 0, 1, 'c')),
                 layers = [ layer1, layer2, maxout, layerMS])

    #mlp_cost
    missing_target_value = -1
    mlp_cost = MLPCost(cost_type='default',
                            missing_target_value=missing_target_value )
    mlp_cost.setup_dropout(input_include_probs= { 'convRect1' : 0.8 }, input_scales= { 'convRect1': 1. })

    #dropout_cost = Dropout(input_include_probs= { 'convRect1' : .8 },
    #                      input_scales= { 'convRect1': 1. })

    #algorithm
    monitoring_dataset = {'validation':ddmValid, 'mini-train':ddmSmallTrain}

    term_crit  = MonitorBased(prop_decrease = 1e-7, N = 100, channel_name = 'validation_objective')

    kp_ada = KeypointADADELTA(decay_factor = 0.95, 
            #init_momentum = 0.5, 
                        monitoring_dataset = monitoring_dataset, batch_size = batch_size,
                        termination_criterion = term_crit,
                        cost = mlp_cost)

    #train extension
    #train_ext = ExponentialDecayOverEpoch(decay_factor = 0.998, min_lr_scale = 0.001)
    #train_ext = LinearDecayOverEpoch(start= 1,saturate= 250,decay_factor= .01)
    #train_ext = ADADELTA(0.95)

    #train object
    train = Train(dataset = ddmTrain,
                  save_path= save_path,
                  save_freq=10,
                  model = MLPerc,
                  algorithm= kp_ada,
                  extensions = [#train_ext, 
                      MonitorBasedSaveBest(channel_name='validation_objective',
                                                     save_path= best_path)#,

#                                MomentumAdjustor(start = 1,#
 #                                                saturate = 25,
  #                                               final_momentum = finMomentum)
  ] )
    train.main_loop()
    train.save()

if __name__=='__main__':
    main()
