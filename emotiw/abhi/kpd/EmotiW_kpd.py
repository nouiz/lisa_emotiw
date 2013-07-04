from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import os
from pylearn2.utils.string_utils import preprocess
from pylearn2.models.mlp import ConvRectifiedLinear, RectifiedLinear, MLP
import csv
import numpy as np
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
from pylearn2.datasets import preprocessing
from Multisoftmax import MultiSoftmax 
import pickle
from MLPCost import MLPCost
from KeypointSGD import KeypointSGD
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.sgd import MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased
from ExponentialDecayOverEpoch import ExponentialDecayOverEpoch
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from test import *
from theano import function
from theano import tensor as T
from pylearn2.models.maxout import Maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms.sgd import LinearDecayOverEpoch
import sys
params = __import__('experiment_' + sys.argv[1] + '_params')

# The number of features in the Y vector
numberOfKeyPoints = 98*2

class Emotiw_FacialKeypoint(DenseDesignMatrix):
    """
    A Pylearn2 Dataset object for accessing the data for the
    Kaggle facial-keypoint-detection contest for the IFT 6266 H13 course.
    """

    def __init__(self, which_set,
                 start=None,
                 stop=None,
                 axes=('b', 0, 1, 'c'),
                 im_size = (96,96, 3),
                 missing_target_value = -1,
                 stdev=0.8):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                   If you are using this on the DIRO filesystem, you
                   can just use the default value. If you are using this
                   at home, you should download the .csv files from
                   Kaggle and set base_path to the directory containing
                   them.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        self.missing_target_value = missing_target_value
        self.stdev = stdev
        files = {'train': ('/data/lisa/data/faces/hdf5/complete_train_x.npy', '/data/lisa/data/faces/hdf5/complete_train_y.npy'),
                 'test': ('/data/lisa/data/faces/hdf5/complete_test_x.npy', '/data/lisa/data/faces/hdf5/complete_test_y.npy')}

        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)

        train_x = np.memmap(files['train'][0], dtype='uint8', mode='r')
        numSamples = len(train_x)/(96*96*3)
        train_x = train_x.reshape((numSamples,(96*96*3)))
        test_x = np.memmap(files['test'][0], dtype='uint8', mode='r')
        numSamples = len(test_x)/(96*96*3)
        test_x = test_x.reshape((numSamples,(96*96*3)))
        X = np.vstack((train_x.view(),test_x.view()))
        del train_x, test_x
        train_y = np.memmap(files['train'][1], dtype='float32', mode='r')
        numSamples = len(train_y)/(numberOfKeyPoints)
        train_y = train_y.reshape((numSamples,numberOfKeyPoints))
        test_y = np.memmap(files['test'][1], dtype='float32', mode='r')
        numSamples = len(test_y)/numberOfKeyPoints
        test_y = test_y.reshape((numSamples,numberOfKeyPoints))
        y = np.vstack((train_y.view(), test_y.view()))
        del train_y, test_y
        self.out_len = im_size[0]+2 if im_size[0] > im_size[1] else im_size[1]+2

        """    
        # (num_examples, num_keypoints, 2)
        y = y.reshape((y.shape[0],y.shape[1]/2,2))      
        if rescale_ratio is not None:
            y = self.rescale_keypoints(y, rescale_ratio)
        y = make_spatial_keypoints(y)"""
            
        if start is not None:
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X.view()[start:stop, :]
            if y is not None:
                y = y.view()[start:stop, :]
            print y.shape

        self.pixels = np.arange(0,self.out_len)
        #y = self.make_targets(y)
        print 'length of total dataset:', y.shape[0]



        view_converter = DefaultViewConverter(shape=[im_size[0], im_size[1], im_size[2]], axes = axes)
        super(Emotiw_FacialKeypoint, self).__init__(X=X, y=y, view_converter=view_converter)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5
        
    def make_targets(self, y):
        # y : (batch_size, num_keypoints):
        # (batch_size, num_keypoints*2, 98)
        Y = np.zeros((y.shape[0], y.shape[1], 98))
        for i in xrange(y.shape[1]):
            Y[:,i,:] = np.where(y[:,i].reshape(y.shape[0],1)!= self.missing_target_value,
                (np.exp(-(y[:,i].reshape(y.shape[0],1)-self.pixels)**2/(2*self.stdev**2)))/(np.sqrt(2*3.14159265359)*self.stdev),
                 self.missing_target_value)
        return Y




def main():

    #creating layers
        #2 convolutional rectified layers, border mode valid
    batch_size = 64
    lr = params.lr
    finMomentum = params.momentum
    maxout_units  = params.units
    num_pcs = params.pieces
    lay1_reg = lay2_reg = maxout_reg = params.norm_reg
    save_path = './models/no_maxout/titan_lr_0.1_btch_64_momFinal_0.9_maxout_2000_4.joblib'
    best_path = './models/no_maxout/titan_bart10_gpu2_best.joblib'
    #save_path = '../models/titan/titan_lr_1.0_btch_64_momFinal_0.9_maxout_2000_4.joblib'
    #best_path = './models/titan/titan_eos1_best.joblib'
    #save_path = './models/lr10/titan_lr_10.0_btch_64_momFinal_0.9_maxout_2000_4.joblib'
    #best_path = './models/lr10/titan_bart10_gpu0_best.joblib'
    #save_path = './eos3/eos3Ind_1_lr_0_0001_btch_32_momFinal_0_9_maxout_1500_3.joblib'
    #best_path = './eos3/eos3Ind_1_best.joblib'
    numBatches = 400000/batch_size
     
    from emotiw.common.datasets.faces.EmotiwKeypoints import EmotiwKeypoints
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

    ddmTrain = EmotiwKeypoints(hack = 'train', preproc='STD')
    ddmValid = EmotiwKeypoints(hack = 'val', preproc='STD')

    


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
    irange = 0
    layer_name = 'multisoftmax'
    layerMS = MultiSoftmax(n_groups=n_groups,irange = 0.05, n_classes=n_classes, layer_name= layer_name)
    
    #setting up MLP
    MLPerc = MLP(batch_size = batch_size,
                 input_space = Conv2DSpace(shape = [96, 96],
                 num_channels = 3),
                 layers = [ layer1, layer2, maxout, layerMS])

    #mlp_cost
    missing_target_value = -1
    mlp_cost = MLPCost(cost_type='default', 
                            missing_target_value=missing_target_value )
    mlp_cost.setup_dropout(input_include_probs= { 'convRect1' : 1.0 },
                           input_scales= { 'convRect1': 1. })

    #dropout_cost = Dropout(input_include_probs= { 'convRect1' : .8 },
    #                      input_scales= { 'convRect1': 1. })

    #algorithm
    monitoring_dataset = {'validation':ddmValid}

    term_crit  = MonitorBased(prop_decrease = 1e-7, N = 100, channel_name = 'validation_objective')

    kpSGD = KeypointSGD(learning_rate = lr, init_momentum = 0.5, 
                        monitoring_dataset = monitoring_dataset, batch_size = batch_size,
                        termination_criterion = term_crit,
                        cost = mlp_cost)

    #train extension
    #train_ext = ExponentialDecayOverEpoch(decay_factor = 0.998, min_lr_scale = 0.001)
    train_ext = LinearDecayOverEpoch(start= 1,saturate= 250,decay_factor= .01)

    #train object
    train = Train(dataset = ddmTrain,
                  save_path= save_path,
                  save_freq=10,
                  model = MLPerc,
                  algorithm= kpSGD,
                  extensions = [train_ext, 
                                MonitorBasedSaveBest(channel_name='validation_objective',
                                                     save_path= best_path),

                                MomentumAdjustor(start = 1,
                                                 saturate = 25,
                                                 final_momentum = finMomentum)] )
    train.main_loop()
    train.save()

if __name__=='__main__':
    main()
