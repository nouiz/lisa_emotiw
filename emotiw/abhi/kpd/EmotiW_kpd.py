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


from pylearn2.utils import serial
import Image

def generateTest(dataset, modelPath, out_path, batch_size = 8):
    model = serial.load(modelPath)
    # use smallish batches to avoid running out of memory
    model.set_batch_size(batch_size)
    # dataset must be multiple of batch size of some batches will have
    # different sizes. theano convolution requires a hard-coded batch size
    m = dataset.X.shape[0]
    extra = batch_size - m % batch_size
    assert (m + extra) % batch_size == 0
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
                                                        dtype=dataset.X.dtype)), axis=0)
    assert dataset.X.shape[0] % batch_size == 0
        
    X = model.get_input_space().make_batch_theano()
    # (batch_size, 30, 98)
    preY = model.fprop(X)
    # (batch_size, 30)
    Y = (T.arange(0,98).dimshuffle('x','x',0)*preY).sum(2)
    f = function([X], Y)

    y = []
        
    for imgIdx in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
        images = []
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))
        ys = f(x_arg.astype(X.dtype))
        for i in range(batch_size):
            images.append(x_arg[i, :,:,:].reshape((96,96,1)))
            im = Image.fromarray(np.uint8(x_arg[i, :,:,:].reshape((96,96))*255 + 127))
            imgrgb = Image.merge('RGB', (im,im,im))
            pixmap = imgrgb.load()
            print 'batch', i
            for j in range(15):
                print 'in', j
                x , y = ys[i,2*j] , ys[i,2*j+1]
                pixmap[int(x), int(y)] = (0,255,0)
            imgrgb.show()
        return
        
    
    y = np.concatenate(y)
    assert y.shape[0] == dataset.X.shape[0]
    # discard any zero-padding that was used to give the batches uniform size
    y = y[:m]

    submission = []
    with open('submissionFileFormat.csv', 'rb') as cvsTemplate:
        reader = csv.reader(cvsTemplate)
        for row in reader:
            submission.append(row)

    mapping = dict(zip(['left_eye_center_x',
                    'left_eye_center_y',
                    'right_eye_center_x',
                    'right_eye_center_y',
                    'left_eye_inner_corner_x',
                    'left_eye_inner_corner_y',
                    'left_eye_outer_corner_x',
                    'left_eye_outer_corner_y',
                    'right_eye_inner_corner_x',
                    'right_eye_inner_corner_y',
                    'right_eye_outer_corner_x',
                    'right_eye_outer_corner_y',
                    'left_eyebrow_inner_end_x',
                    'left_eyebrow_inner_end_y',
                    'left_eyebrow_outer_end_x',
                    'left_eyebrow_outer_end_y',
                    'right_eyebrow_inner_end_x',
                    'right_eyebrow_inner_end_y',
                    'right_eyebrow_outer_end_x',
                    'right_eyebrow_outer_end_y',
                    'nose_tip_x',
                    'nose_tip_y',
                    'mouth_left_corner_x',
                    'mouth_left_corner_y',
                    'mouth_right_corner_x',
                    'mouth_right_corner_y',
                    'mouth_center_top_lip_x',
                    'mouth_center_top_lip_y',
                    'mouth_center_bottom_lip_x',
                    'mouth_center_bottom_lip_y'], range(30)))
    '''
    for row in submission[1:]:
        imgIdx = int(row[1]) - 1
        keypointName = row[2]
        keyPointIndex = mapping[keypointName]
        row.append(y[imgIdx, keyPointIndex])

    with open(out_path, 'w') as cvsTemplate:
        writer = csv.writer(cvsTemplate)
        for row in submission:
            writer.writerow(row)
    '''
  

def test_works():
    load = True

    if load == False:
        from emotiw.common.datasets.faces.EmotiwKeypoints import EmotiwKeypoints
        ddmTrain = EmotiwKeypoints(start=0, stop =40000)
        ddmValid = EmotiwKeypoints(start= 40000)
      #  ddmTest =  EmotiwKeypoints(start=0, stop = )
        # valid can_fit = false
        #pipeline = preprocessing.Pipeline()

        #stndrdz = preprocessing.Standardize()
        #stndrdz.apply(ddmTrain, can_fit=True)
        
        #doubt, how about can_fit = False?
        #stndrdz.apply(ddmValid, can_fit=False)
        #stndrdz.apply(ddmTest, can_fit=False)

        GCN = preprocessing.GlobalContrastNormalization()
        GCN.apply(ddmTrain, can_fit =True)
        GCN.apply(ddmValid, can_fit =False)
        #GCN.apply(ddmTest, can_fit =False)
    
        pcklFile = open('emotiw_kpd.pkl', 'wb')
        obj = (ddmTrain, ddmValid, ddmTest, GCN, stndrdz)
        pickle.dump(obj, pcklFile)
        pcklFile.close()
        return
    elif False:
        pcklFile = open('emotiw_kpd.pkl', 'rb')
        (ddmTrain, ddmValid, ddmTest, GCN, stndrdz) = pickle.load(pcklFile)
        pcklFile.close()
        batch_size = 8
        print 'going to compute test error'
        generateTest(ddmTrain, 'kpd_maxout_best.pkl', 'output_maxout2pcs.csv')
        return

    #creating layers
        #2 convolutional rectified layers, border mode valid
    batch_size = 64
    lr = 0.0001
    finMomentum = 0.8
    maxout_units  = 2000
    num_pcs = 3
    save_path = './titan/titanInd_1_lr_0_0001_btch_64_momFinal_0_8_maxout_2000_3.joblib'
    best_path = './titan/titanInd_1_best.joblib'
    #save_path = './eos3/eos3Ind_1_lr_0_0001_btch_32_momFinal_0_9_maxout_1500_3.joblib'
    #best_path = './eos3/eos3Ind_1_best.joblib'
    numBatches = 400000/batch_size
    print 'Apllying preprocessing' 
    from emotiw.common.datasets.faces.EmotiwKeypoints import EmotiwKeypoints
    '''
    ddmTrain = EmotiwKeypoints(start=0, stop =40000)
    ddmValid = EmotiwKeypoints(start=40000, stop = 44000)
    ddmTest = EmotiwKeypoints(start=44000)
    
    stndrdz = preprocessing.Standardize()
    stndrdz.applyLazily(ddmTrain, can_fit=True, name = 'train')
    print 'preprocess STD 1'
    stndrdz.applyLazily(ddmValid, can_fit=False, name = 'val')
    print 'precprocessed std2'
    stndrdz.applyLazily(ddmTest, can_fit=False, name = 'test')
    print 'making gcn object'
    GCN = preprocessing.GlobalContrastNormalization(batch_size = 1000)
    print 'aplying gcn to ddmTrain'
    GCN.apply(ddmTrain, can_fit =True, name = 'train')
    print 'precrocessed gcn1'
    GCN.apply(ddmValid, can_fit =False, name = 'val')
    print 'precrocessed gcn1'
    GCN.apply(ddmTest, can_fit = False, name = 'test')
    return
    '''

    ddmTrain = EmotiwKeypoints(hack = 'train')
    #print ddmTrain.y
    ddmValid = EmotiwKeypoints(hack = 'val')
    #print ddmValid.y
    #print ddmTrain.y
    #print ddmTrain.y.shape
    layer1 = ConvRectifiedLinear(layer_name = 'convRect1',
                     output_channels = 64,
                     irange = .05,
                     kernel_shape = [5, 5],
                     pool_shape = [3, 3],
                     pool_stride = [2, 2],
                     max_kernel_norm = 1.9365)
    layer2 = ConvRectifiedLinear(layer_name = 'convRect2',
                     output_channels = 64,
                     irange = .05,
                     kernel_shape = [5, 5],
                     pool_shape = [3, 3],
                     pool_stride = [2, 2],
                     max_kernel_norm = 1.9365)

        # Rectified linear units
    #layer3 = RectifiedLinear(dim = 3000,
    #                         sparse_init = 15,
    #                 layer_name = 'RectLin3')

    #Maxout layer
    maxout = Maxout(layer_name= 'maxout',
                    irange= .005,
                    num_units= maxout_units,
                    num_pieces= num_pcs,
                    max_col_norm= 1.9)


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
    mlp_cost.setup_dropout(input_include_probs= { 'convRect1' : .8 },
                           input_scales= { 'convRect1': 1. })

    #dropout_cost = Dropout(input_include_probs= { 'convRect1' : .8 },
    #                      input_scales= { 'convRect1': 1. })

    #algorithm
    
    # learning rate, momentum, batch size, monitoring dataset, cost, termination criteria
#monitoring_dataset = {'validation':ddmValid, 'training': ddmTrain}
    term_crit  = MonitorBased(prop_decrease = 0.00001, N = 30, channel_name = 'validation_objective')
    kpSGD = KeypointSGD(learning_rate = lr, init_momentum = 0.5, 
                        monitoring_dataset = {'validation':ddmValid}, batch_size = batch_size,
                        termination_criterion = term_crit,
                        cost = mlp_cost)

    #train extension
    train_ext = ExponentialDecayOverEpoch(decay_factor = 0.998, min_lr_scale = 0.01)
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
    test_works()
