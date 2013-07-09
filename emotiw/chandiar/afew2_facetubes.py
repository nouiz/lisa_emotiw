# Based on Mehdi's afew2_facetubes.py from emotiw/scripts/mirzamom/conv3d

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
import cv, cv2
import warnings

class AFEW2FaceTubes(DenseDesignMatrix):
    def __init__(self, which_set, sequence_length = 3, preload_facetubes=True,
                 batch_size = None, preproc=[], size=(96, 96), 
                 greyscale = False):

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

        dataset = AFEW2ImageSequenceDataset(preload_facetubes=False, 
                                            preproc=preproc, 
                                            size=size)

        self.dataset = dataset

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
                    temp = []
                    for frame in ft:
                        if greyscale:
                            temp.append(cv2.cvtColor(frame, cv.CV_BGR2GRAY))
                        else:
                            temp.append(frame)
                    ft = numpy.array(temp)
                    _features.append(ft)
                    _clip_ids.append(idx)
                    _targets.append(tgt)

            features = []
            #self.clip_ids = []
            targets = []
            count = 0
            for feat, clip_id, target in zip(_features, _clip_ids, _targets):
                # duplicate frames at the end if it's not modulo of sequence_length
                modulo = feat.shape[0] % sequence_length
                #if modulo != 0:
                    # TODO return a warning here
                    #feat = numpy.concatenate((feat, feat[-modulo,:,:,:][None,:,:,:]))
                for i in xrange(feat.shape[0] - sequence_length + 1):
                    features.append(feat[i:i+sequence_length,:,:])
                    assert len(features[-1]) == sequence_length
                    count += 1
                    #self.clip_ids.append(clip_id)
                    targets.append(target)

            self.n_samples = count
            feat_shape = features[0].shape
            features = numpy.concatenate(features)
            features = features.reshape((self.n_samples, sequence_length * numpy.product(feat_shape[1:])))

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
        super(AFEW2FaceTubes, self).__init__(X = features, y = targets, axes = ('b', 'c', 0, 1))

if __name__ == '__main__':
    # Load the unprocessed train face tubes dataset.
    #train = AFEW2FaceTubes('train', sequence_length = 1)
    #mehdi_bbox_coords = train.dataset.get_bbox_coords(0)

    # Load the smoothed train and valid face tubes of size 48 x 48 and remove
    # background faces as many as possible.
    print '... loading smooth face tubes'
    smooth_train = AFEW2FaceTubes('train', sequence_length = 1, size=(48, 48),
        preproc=['smooth', 'remove_background_faces'], greyscale=False)
    raul_bbox_coords = smooth_train.dataset.get_bbox_coords(0)
    

    import pdb; pdb.set_trace()
