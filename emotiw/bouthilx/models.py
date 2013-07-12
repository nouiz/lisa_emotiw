from pylearn2.utils import serial
import pylearn2.datasets.tfd as tfd
from theano import config
from theano import function
import numpy as np
import os

class MyModel:
    def __init__(self,model_path,dataset_stats_path):
        self.model = serial.load(model_path)
        X = self.model.get_input_space().make_batch_theano()
        # load mean and std saved
        dataset_stats = np.load(dataset_stats_path)
        mean = np.cast[config.floatX](dataset_stats['arr_0'])
        std = np.cast[config.floatX](dataset_stats['arr_1'])
        Y = self.model.fprop((X-mean)/(np.cast[config.floatX](1e-4)+std))
        self._fprop = function([X],Y)
        self.batch_size = self.model.get_test_batch_size()

    def fprop(self,X):
        """
            dimensions have to be in order (batch_size,48,48)
        """

        # if X is a single image flattened or not
        if tuple(X.shape) == (48,48) or len(X.shape)==1:
            X = X.reshape((1,48,48,1))
        else:
            X = X.reshape((X.shape[0],48,48,1))

        y_hat = []
        for idx in xrange(0,X.shape[0],self.batch_size):
            if idx+self.batch_size > X.shape[0]:
                batch = np.cast[config.floatX](np.zeros([self.batch_size]+list(X.shape[1:])))
                batch[:X.shape[0]-idx] = X[idx:X.shape[0]]
            else:
                batch = X[idx:idx+self.batch_size]
            y_hat.append(self._fprop(batch))

        y_hat = np.concatenate(y_hat,0)
        # cut-off extra 0s
        y_hat = y_hat[:X.shape[0]]
            
        return y_hat

def test(model_path,dataset_stats_path):
#    model = serial.load(model_path)
    if not os.path.isfile(dataset_stats_path):
        dataset = tfd.TFD(
            which_set='train',
            image_size=48,
            one_hot= True)
        arr = dataset.get_design_matrix()
        arr = arr.reshape((arr.shape[0],48,48,1))
        print arr.mean(0).shape
        print arr.shape
        np.savez(dataset_stats_path,arr.mean(0),arr.std(0))
    model = MyModel(model_path,dataset_stats_path)
    dataset = tfd.TFD(
        which_set='valid',
        image_size=48,
        one_hot= True)

    batch_size = 32
    data = dataset.get_design_matrix()
    targets = dataset.get_targets()
    y_hat = model.fprop(data)
    print y_hat.shape
    print data.shape
    print np.mean(y_hat.argmax(1)==targets.argmax(1))

if __name__=="__main__":
    test("/data/lisa/exp/best_models/emotiw/static_faces/xavier_conv_net.pkl",
         "/data/lisa/exp/best_models/emotiw/static_faces/xavier_data_stats.npz")
