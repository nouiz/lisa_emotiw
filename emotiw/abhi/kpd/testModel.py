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
from pylearn2.utils import serial
import Image

def generateTest(dataset, modelPath,  batch_size = 8):
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
    Y = (T.arange(0,96).dimshuffle('x','x',0)*preY).sum(axis = 2)
    #Y = T.argmax(preY, axis=2)
    f = function([X], Y)

    y = []
        
    for imgIdx in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[imgIdx * batch_size:(imgIdx + 1) * batch_size, :]
        images = []
        if X.ndim > 2:
            x_arg2 = dataset.get_topological_view(x_arg)
        y.append(f(x_arg2.astype(X.dtype)))
        ys = f(x_arg2.astype(X.dtype))
        for i in range(batch_size):
            #print ys[i, 25,:]
            images.append(x_arg2)
            transf = x_arg[i]
            diff = (max(transf) - min(transf))
            transf = (transf/diff)*255
            transf = np.cast['uint8'](transf)
            im = Image.frombuffer(data=transf, size=(96,96), mode='RGB')
            #im.show()
            #continue
            im = im.convert('L')
            im = im.convert('RGB')
            
            #imgrgb = Image.merge('RGB', (im,im,im))
            pixmap = im.load()
            print 'batch', i
            for j in range(numberOfKeyPoints/2):
                #print 'in', j 
                #x, y = (0,0)
                x , y = ys[i,2*j] , ys[i,2*j+1]
                pixmap[int(x), int(y)] = (0,255,0)
            im.save('./results/'+str(i)+'.png')
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

  

def test_works():
    
    #creating layers
        #2 convolutional rectified layers, border mode valid
    #model_file = 'eos3/eos3Ind_1_best.joblib'
    model_file = 'titan2/titan2_lr_0.01_finMomentum_0.8_batch_size_32_mlp.RectifiedLinear.joblib'
    model_file = 'titan2/titan2_best.joblib'
    model_file = 'titan3/titan3_best.joblib'
    model_file = '/u/zumerjer/Documents/lisa_emotiw/emotiw/zumerjer/models/bart10_no_preproc.joblib'

    batch_size = 64
    from emotiw.zumerjer.comboDS import ComboDS
    
    ddmTest = ComboDS(path='/Tmp/zumerjer/all_train_test.h5')

    generateTest(ddmTest, model_file, batch_size = batch_size)

if __name__=='__main__':
    test_works()
