from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

import emotiw.common.datasets.faces.afew2 as afew2
import numpy as np
from collections import defaultdict

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

        ind = np.arange(self.X.shape[0])
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]

        super(AFEWStatsDataset, self).__init__(X=X, y=y, view_converter=view_converter)



class AFEWDataset(DenseDesignMatrix):

    def __init__(self,which_set,base_path, 
            start = None,
            stop = None,
            preprocessor = None,
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c'),
            fit_test_preprocessor = False):

        print base_path,"?"

        self.test_args = locals()
        self.test_args['which_set'] = 'public_test'
        self.test_args['fit_preprocessor'] = fit_test_preprocessor
        del self.test_args['start']
        del self.test_args['stop']
        del self.test_args['self']


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

        X = X.reshape((X.shape[0],96*96*3))

        view_converter = DefaultViewConverter(shape=[96,96,3], axes=axes)

        super(AFEWDataset, self).__init__(X=X, y=y, view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)

def build_face_datasets():
    d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)
    targets = defaultdict(list)
    faces = defaultdict(list)
    for i, [clip, info, target] in enumerate(zip(d.imagesequences,d.seq_info,d.labels)):
        
        for facetube in d.get_facetubes(i):
            onehot = np.zeros(len(d.emotionNames.keys()))
            onehot[d.emotionNames.values().index(target)] = 1.0
            targets[info[0]] += [onehot]*facetube.shape[0]
            faces[info[0]] += [face for face in facetube]
            
#        print i, len(targets[info[0]]), target

    for name, split in faces.items():

        ind = np.arange(len(split))
        np.random.shuffle(ind)

        X = np.array(split)
        y = np.array(targets[name])
        X = X[ind]
        y = y[ind]
        np.save("/data/lisatmp/bouthilx/%s_X.npy" % name,X)
        np.save("/data/lisatmp/bouthilx/%s_y.npy" % name,y)

if __name__ == "__main__":
    build_face_datasets()
