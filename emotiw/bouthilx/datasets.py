from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

import emotiw.common.datasets.faces.afew2 as afew2
import numpy as np
from collections import defaultdict
import emotiw.bouthilx.timer as timer
import os
import PIL.Image as Image

classes = ['anger','disgust','fear','happy','sad','surprise','neutral']

def one_hot(y):
    if len(y.shape)==1:
        tmp_y = np.zeros((y.shape[0],7))
        for i,t in enumerate(y):
            tmp_y[i,t] = 1.0
        y = tmp_y

    return y

class FeaturesDataset(DenseDesignMatrix):
    def __init__(self,features_paths,targets_path,
                      base_path="",
                      normalize=True,
                      seed=193847,
                      shuffle=True):
        X = []
        for f in features_paths:
            X.append(np.load(os.path.join(base_path,f)))
        np.random.seed(seed)
        X = np.concatenate(X,axis=1)

        if normalize:
            # set 0s to mean
            tmp = X[:]
            tmp -= tmp.min(0)
            tmp /= (tmp==0) + tmp.max(0)
            # range is [-1,1]
            tmp = tmp*2.0-1.0 
            tmp = (X.min(0)!=X.max(0))*tmp # set empty dimensions to 0
            X = tmp

#        print X.mean(0)
#        print X.min(0)
#        print X.max(0)
#        print X.min(0)==X.max(0)
#        import sys
#        
#        sys.exit(0)

        y = np.load(os.path.join(base_path,targets_path))
        if len(y.shape)==1:
            tmp_y = np.zeros((y.shape[0],7))
            for i,t in enumerate(y):
                tmp_y[i,t] = 1.0
            y = tmp_y

        if shuffle:
            ind = np.arange(X.shape[0])
            np.random.shuffle(ind)
            X = X[ind]
            y = y[ind]

        super(FeaturesDataset, self).__init__(X=X, y=y)

class AFEWStatsDataset(DenseDesignMatrix):

    def __init__(self,which_set,base_path):
        if which_set == "train":
            X = np.load(base_path+"/Train_x.npy")
            y = np.load(base_path+"/Train_y.npy")
        elif which_set == "valid":
            X = np.load(base_path+"/Val_x.npy")
            y = np.load(base_path+"/Val_y.npy")
        else:
            raise ValueError("Unrecognized dataset name: " + which_set)

        ind = np.arange(X.shape[0])
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]

        super(AFEWStatsDataset, self).__init__(X=X, y=y)

class AFEWDataset(DenseDesignMatrix):

    def __init__(self,which_set,base_path,
            start = None,
            stop = None,
            preprocessor = None,
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c')):

        if which_set == "train":
            X = np.load(base_path+"/Train_X.npy")
            y = np.load(base_path+"/Train_y.npy")
        elif which_set == "valid":
            X = np.load(base_path+"/Val_X.npy")
            y = np.load(base_path+"/Val_y.npy")
        else:
            raise ValueError("Unrecognized dataset name: " + which_set)

        if start is not None:
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop]
            y = y[start:stop]

#        X = X.transpose(0,3,1,2).reshape((X.shape[0],96*96*3))
        X = X.reshape((X.shape[0],96*96*3))

        view_converter = DefaultViewConverter(shape=[96,96,3], axes=axes)

        super(AFEWDataset, self).__init__(X=X, y=y, view_converter=view_converter,
                                          preprocessor=preprocessor,fit_preprocessor=fit_preprocessor)

def build_face_datasets(url,p=1.0):
    d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)
    targets = defaultdict(list)
    faces = defaultdict(list)
    t = timer.Timer(len(d.labels),min_time=1)
    t.start()
    for i, [clip, info, target] in enumerate(zip(d.imagesequences,d.seq_info,d.labels)):
        for facetube in d.get_facetubes(i):
            facetube = np.array([facetube[j] for j in xrange(0,len(facetube),int(1/p))]) # hop hop
            onehot = np.zeros(len(d.emotionNames.keys()))
            onehot[classes.index(target)] = 1.0
            targets[info[0]] += [onehot]*facetube.shape[0]
            faces[info[0]] += list(facetube)

#        print i, len(targets[info[0]]), target
        t.print_update(1)
    t.over()

    print "Saving images"

    for name, split in faces.items():

        ind = np.arange(len(split))
        np.random.shuffle(ind)

        X = np.array(split)
        y = np.array(targets[name])
        X = X[ind]
        y = y[ind]
        np.save("%s%s_X.npy" % (url, name),X)
        np.save("%s%s_y.npy" % (url, name),y)

def process_stats(X):
    """
        mean saturation
        mean illumination
        mean red
        mean green
        mean blue
    """

    return np.array([1.0,1.0,1.0,1.0,1.0])

    # RGB
    m = np.min(X,2).T
    M = np.max(X,2).T
    C = M-m
    Cmsk = C!=0

    I = M
    S = np.zeros(X.T[0].shape)
    S[Cmsk] = ((255*C)/I)[Cmsk]

    R = X[:,:,0].mean()
    G = X[:,:,1].mean()
    B = X[:,:,2].mean()

    return np.array([S.mean(),I.mean(),R,G,B])

def build_stats_dataset(url):
    d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)
    targets = defaultdict(list) # for Train and Val splits
    clips = defaultdict(list)
    t = timer.Timer(len(d.labels),min_time=1)
    t.start()
    for i, [clip, info, target] in enumerate(zip(d.imagesequences,d.seq_info,d.labels)):

        sequence = []
#        for image in clip:
#            sequence.append(np.asarray(Image.open(image.original_image_path).convert('RGB'),int))

        onehot = np.zeros(len(d.emotionNames.keys()))
        onehot[classes.index(target)] = 1.0
        targets[info[0]].append(onehot)
        clips[info[0]].append(process_stats(np.array(sequence)))
        print clips[info[0]][-1]
        print targets[info[0]][-1]
#        print i, len(targets[info[0]]), target
        t.print_update(1)
    t.over()

    print "Saving images"

    for name, split in clips.items():

#        ind = np.arange(len(split))
#        np.random.shuffle(ind)

        X = np.array(split)
        y = np.array(targets[name])
#        X = X[ind]
#        y = y[ind]
        np.save("%s%s_X.npy" % (url, name),X)
        np.save("%s%s_y.npy" % (url, name),y)

if __name__ == "__main__":
    url = "/data/lisatmp/bouthilx/facetubes/p10/"
#    build_face_datasets(url,p=0.10)
    url = "/data/lisatmp/bouthilx/stats/"
    build_stats_dataset(url)
