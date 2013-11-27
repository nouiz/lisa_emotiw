import os
import random
import numpy as np
import sys

import Queue
import threading
import time

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from video_tools import *
from hollywood2_tools import *
import Inference

from pca import pca, whiten

#sys.path.insert(0,
#    os.path.join(
#        os.path.realpath(os.path.abspath('.')),
#        'libsvm-3.13/python'))
sys.path.append('/data/lisa/exp/faces/emotiw_final/Kishore/inference_line/libsvm-3.13/python')

import svmutil
from chi2 import chi2


def onehot(x, numclasses=None):
    """
    Function to compute one hot vectors given set of values.
    @prams
    x         : array of values
    numclasses: total number of classes
    @returns
    result    : one hot vector

    Example
      x = [1,3], numclasses=5
      result = [[1,0,0,0,0]
                [0,0,1,0,0]]
    """

    if x.shape == ():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses])
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x == c)] = 1
        result[..., c] += z
    return result

def compute_kernel_matrices(train_inputs, test_inputs):
    """Computes the kernel matrices for train and test inputs
    Args:
        train_inputs: training inputs (numpy array, stacked row-wise)
        test_inputs: test inputs (numpy array, stacked row-wise)
    Returns:
        tuple of kernel matrices (one for train and one for test inputs)
    """
    kernel_train = chi2.chi2_kernel(train_inputs,train_inputs)
    kernel_train = np.concatenate((np.arange(1,kernel_train[0].shape[0]+1).reshape(-1,1), kernel_train[0]),1)

    kernel_test = chi2.chi2_kernel(test_inputs,train_inputs)
    kernel_test = np.concatenate((np.arange(1,kernel_test[0].shape[0]+1).reshape(-1,1), kernel_test[0]),1)
    return (kernel_train, kernel_test)

def model_initializer(model_file):
    """
    Function to intialize an intance of Inference model
    and load the parameters.
    paramters passed to Inference model nvis and nmap 
    based on the paper.
    @params   
    model_file: Name of npy or npz file with Inference 
                model parameters
    @returns
    model     : Instance of a model with paramteres loaded
    """

    numpy_rng  = np.random.RandomState(1)
    theano_rng = RandomStreams(1)   
    model = Inference.Inference(numpy_rng=numpy_rng, theano_rng=theano_rng, 
                            nvis=2560, npro=300)
    model.load(model_file)

    return model


class ActionRecognizer(object):
    """
    Class for predicting activity labels for a given list videos.
    Each video can be processed parallely.
    """
    def __init__(self,args):
        """        
        @params
        args   : A dictionary of requirements for loading precomputed parameters.
        centroids_file    : name of npy file with kmeans centriods for vector quantization
        nthreads  :Number of cpu threads for parallel computation. Each thread handles a videoself.
        model_file : npz pr npy parameter file with weight matrix 
        (combined whitening anf filter weights), pixel wise mean and std.      
        videos_path   : path to actual location of videos in th list.
        classifier    : pretrained classifier model file
        clip_ids      : IDs of the clips to process this time
        """                
        
        self.centroids = np.load(args['centroids_file'])
        self.nthreads = args['nthreads']
        
        # list of model instances used by each thread.
        self.model_instances = [model_initializer(args['model_file']) for i in range(self.nthreads)]

        clip_ids = args['clip_ids']
        if len(clip_ids) == 0:
            self.filelist = os.listdir(args['videos_path'])
        else:
            self.filelist = ['%s.avi' % clip_id for clip_id in clip_ids]
        for i in range(len(self.filelist)):
            self.filelist[i] = os.path.join(args['videos_path'], self.filelist[i])
        self.train_inputs = np.load(args['train_data_file'])['inputs']
        self.train_outputs = np.load(args['train_data_file'])['outputs']
        #print self.filelist



    def get_dense_samples(self):

        """
        generates dense samples and processes them into video descriptor.
        """

        nthreads = self.nthreads

        start = time.time()

        numvis0 = 2560
        numvis1 = 100

        # squared sum of kmeans centroids for quantization of computed feature vectors
        c2 = 0.5 * np.sum(self.centroids ** 2, axis=1).reshape((-1,1))
        
        inputs = np.empty((len(self.filelist), self.centroids.shape[0]), dtype=np.float32)

        jobs = Queue.Queue()
        for i in range(len(self.filelist)):
            jobs.put(i)

        input_mutex = threading.Lock()

        def assign_centroid_indices(modelidx):
            """Samples, preprocesses, gets mappings and assigns cluster centers
            @params
            modelidx: index of a model instance
            @returns:
            Video descriptor (histogram of feature vectors allocated to kmeans centroids)
            """
            while True:
                try:
                    job = jobs.get_nowait()
                    print 'processing file {}/{}'.format(job + 1, len(self.filelist))
                except Queue.Empty:
                    return

                features = sample_clips_dense(video=self.filelist[job],
                                              framesize=16,
                                              horizon=10,
                                              temporal_subsampling=False,
                                              )
                features_sb = np.zeros((features.shape[0],300),dtype=theano.config.floatX)
                MM = np.zeros((features.shape[0],1),dtype=theano.config.floatX)


                batchsize = 5000
                ncases = features.shape[0]
                nbatches = (ncases - 1) / batchsize + 1
                
                for bidx in xrange(nbatches):
                    start = bidx * batchsize
                    end = min((bidx + 1) * batchsize, ncases)
                    
                    # Computing feature vectors for blocks
                    features_tmp = self.model_instances[modelidx].products_batchwise(features[start:end],1000)
                    MM_tmp = ((features_tmp-0.5)/0.5).sum(1).reshape(-1,1)/300.
                    
                    features_sb[start:end]=features_tmp
                    MM[start:end]=MM_tmp

                features = features_sb
                
                # Allocation to kmeans centroids and histogram computaion
                c3 = 0.5 * np.sum(features**2,axis=1).reshape((1,-1))
                c3 = c2 + c3
                features = onehot(np.argmin(c3 - np.dot(self.centroids, features.T), axis=0), self.centroids.shape[0])*MM
                features = features.sum(0).reshape((1, -1))
                input_mutex.acquire()
                inputs[job, :] = features / np.float32(np.sum(features))
                input_mutex.release()
    
        threads = []
        for threadidx in range(nthreads):
            # create some lock so this function stays alive until the threads are
            # finished
            threads.append(threading.Thread(target=assign_centroid_indices, args=(threadidx, )))
            threads[-1].start()
        while True:
            all_finished = True
            for threadidx in range(len(threads)):
                if threads[threadidx].is_alive():
                    all_finished = False
            if all_finished:
                print 'all threads finished'
                break
            else:
                time.sleep(3)
    
        print 'time spent in get_dense_samples(): {0:d}'.format(int(time.time() - start))

        inputs = np.vstack(inputs).astype(np.float32)

        self.inputs = inputs
        np.savez('test_data.npz',inputs=self.inputs,outputs=np.ones(len(self.filelist),))

    def classify(self,args):
        """
        classifies the computed video descriptors into one of the activity classes.
        """

        test_inputs = np.load('test_data.npz')['inputs']
        test_labels = np.load('test_data.npz')['outputs'] #dummy labels (all are ones)

        k_train, k_test = \
                          compute_kernel_matrices(self.train_inputs, test_inputs)

        m = svmutil.svm_train(self.train_outputs.tolist(),k_train.tolist(),' -t 4 -q -s 0 -b 1 -c %f'%100)
        pl, pac, p_val  = svmutil.svm_predict(test_labels.tolist(), k_test.tolist(),m,'-b 1') 
        
        m_labels = svmutil.get_labels(m)
    
        probs = np.array(p_val)[:,np.array(m_labels).argsort()]
        probs_crr = np.concatenate((probs[:,0:4],probs[:,5:7],probs[:,4:5]),1)

        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            
        #flist = os.listdir(args['videos_path'])
        flist = self.filelist

        f = open('activity_recognition_test_results.txt','wb')

        probs_crr = np.around(probs_crr,decimals=3)
        pl_crr = probs_crr.argmax(1)

        for i in range(len(flist)):
            f.write(flist[i][:-4]+'\t'+labels[pl_crr[i]]+'\t'+'\t'.join(map(str,probs_crr[i])) +'\n')
        
        f.close()
        


if __name__ == '__main__':

    #args = { 'nthreads' : 2,
    #         'centroids_file' : 'kmeans_centroids.npy',
    #         'model_file' : 'model_params.npz',                    
    #         'videos_path'   :  '/data/lisa/exp/faces/emotiw_final/Kishore/inference_line/Test_Data/',             
    #         'train_data_file': 'chal_train_data.npz'
    #        }

    args = {'nthreads': 2,
            'centroids_file': sys.argv[1],
            'model_file': sys.argv[2],
            'videos_path': sys.argv[3],
            'train_data_file': sys.argv[4],
            'clip_ids': sys.argv[5:]
            }

    # Initializing the ActionRecognizer class
    Recog = ActionRecognizer(args)
    # Computing the video descriptors
    Recog.get_dense_samples()
    # pridicting labels for computed video descriptors
    Recog.classify(args)

# vim:et ts=4 sw=4 sts=4 expandtab:
