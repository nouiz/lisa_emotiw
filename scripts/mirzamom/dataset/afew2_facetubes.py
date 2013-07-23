"""
This file defines a Pylearn2 Dataset containing the face-tubes data for afew2

All the facetubes for a given clip will be returned as different examples,
each labeled with the label for the full clip.
"""
# In-house dependencies
from pylearn2.space import CompositeSpace, VectorSpace

# Current project
from emotiw.common.datasets.faces.afew2 import AFEW2ImageSequenceDataset
from emotiw.common.datasets.faces.faceimages import basic_7emotion_names
from emotiw.common.datasets.faces.facetubes import FaceTubeDataset, FaceTubeSpace
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy
import warnings

class AFEW2FaceTubes(DenseDesignMatrix):
    def __init__(self, which_set, sequence_length = 3, preload_facetubes=True, batch_size = None):
        if which_set == 'train':
            which_set = 'Train'
        elif which_set == 'valid':
            which_set = 'Val'
        if which_set not in ['Train', 'Val']:
            raise ValueError(
                "Unrecognized value for 'which_set': %s. "
                "Valid values are 'Train' and 'Val'." % which_set)

        if not preload_facetubes:
            raise NotImplementedError(
                "For now, we need to preload all facetubes")

        dataset = AFEW2ImageSequenceDataset(preload_facetubes=False)
        train_idx, val_idx = dataset.get_standard_train_test_splits()[0]
        if which_set == 'Train':
            data_idx = train_idx
        elif which_set == 'Val':
            data_idx = val_idx
        else:
            raise AssertionError

        if preload_facetubes:
            _features = []
            _clip_ids = []
            _targets = []

            for idx in data_idx:
                fts = dataset.get_facetubes(idx)
                tgt = basic_7emotion_names.index(dataset.get_label(idx))
                for ft in fts:
                    _features.append(ft)
                    _clip_ids.append(idx)
                    _targets.append(tgt)

            import ipdb
            ipdb.set_trace()

            features = []
            #self.clip_ids = []
            targets = []
            for feat, clip_id, target in zip(_features, _clip_ids, _targets):
                # duplicate frames at the end if it's not modulo of sequence_length
                modulo = feat.shape[0] % sequence_length
                if modulo != 0:
                    # TODO reuturn a warning here
                    feat = numpy.concatenate((feat, feat[-modulo,:,:,:][None,:,:,:]))
                for i in xrange(feat.shape[0] / sequence_length):
                    features.append(feat[i:i+sequence_length,:,:,:])
                    #self.clip_ids.append(clip_id)
                    targets.append(target)

            self.n_samples = len(features)
            feat_shape = features[0].shape
            features = numpy.concatenate(features)
            features = features.reshape((self.n_samples, sequence_length * feat_shape[1] * feat_shape[2] * feat_shape[3]))

        one_hot = numpy.zeros((self.n_samples, 7), dtype = 'float32')
        for i in xrange(self.n_samples):
            one_hot[i, targets[i]] = 1.
        targets = one_hot

        if batch_size is not None and self.n_samples % batch_size != 0:
            warnings.warn("since batch size is forced adding some duplicate data, be carefull when comparing results. fixed batch size is needed usually for convolution networks")
            self.n_samples = self.n_samples - (self.n_samples % batch_size)
            features = features[:self.n_samples]
            targets = targets[:self.n_samples]

        #view_converter = dense_design_matrix.DefaultViewConverter((sequence_length, 96, 96, 3), axes = ('b', 't', 0, 1, 'c'))
        super(AFEW2FaceTubes, self).__init__(X = features, y = targets, axes = ('b', 't', 0, 1, 'c'))

if __name__ == '__main__':
    data = AFEW2FaceTubes('train', sequence_length = 3)
    data = AFEW2FaceTubes('train', sequence_length = 4)

