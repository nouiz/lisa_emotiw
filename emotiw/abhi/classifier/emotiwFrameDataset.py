import numpy as np
import functools

#from emotiw.common.datasets.faces.facetubes import FaceTubeSpace
from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import resolve_iterator_class
from emotiw.common.datasets.faces.faceimages import basic_7emotion_names

import os


class EmotiwFrameIterator(object):
    def __init__(self, dset, mode, data_specs, return_tuple, convert_fns, rng=None):
        self.dset = dset
	self.stochastic = False
	self.num_examples = dset.x.shape[0]
        self.mode =mode
        self.rng = rng
        self.return_tuple = return_tuple
        self.data_specs = data_specs
        space, source = data_specs
        assert source == ('features', 'targets')
        self.convert_fns = convert_fns
    
    def __len__(self):
        return self.dset.n_samples

    def __iter__(self):
        return self

    def next(self):
        next_idx = self.mode.next()
        feature = self.dset.x[next_idx]
        in_feature_space = self.dset.space.components[0]
        out_feature_space = self.data_specs[0].components[0]
        if in_feature_space != out_feature_space:
            feature = in_feature_space.np_format_as(feature, out_feature_space)
        target = self.dset.y[next_idx]

        if self.dset.one_hot:
            one_hot = np.zeros((len(target),7,), dtype=np.float32)
            one_hot[target] = 1.
            target = one_hot

        return (feature, target)


class EmotiwFrameDataset(Dataset):
    def __init__(self, which_set,
                 one_hot=False,
                 shuffle_rng=None, 
                 size=(48,48), num_channels=1,
                 splitRatio = 0.7,
                 path=None):

        if path is None:
            #path = '/data/lisa/data/faces/EmotiW/preproc/arranged_data'
            path = '/Tmp/aggarwal/arranged_data'

        self.x = np.memmap(path + '_x.npy', mode='r', dtype='float32')
        self.y = np.memmap(path + '_y.npy', mode='r', dtype='uint8')
        self.y = self.y.view()
        self.y.shape = (len(self.y), 1)

        self.x = self.x.view()
        self.x.shape = (len(self.y), size[0], size[1], num_channels)
        

        numSamples = self.x.shape[0]
        if which_set =='train':
            start = 0
            stop = int(numSamples*splitRatio)
        elif which_set == 'val':
            start =  int(numSamples*splitRatio)
            stop = numSamples
            
        self.x = self.x[start:stop]
        self.y =self.y[start:stop]
        
        if shuffle_rng is None:
            shuffle_rng = np.random.RandomState((2013, 06, 11))
        elif not isinstance(shuffle_rng, np.random.RandomState):
            shuffle_rng = np.random.RandomState(shuffle_rng)

        self.permutation = shuffle_rng.permutation(len(self.y))
        self.one_hot = one_hot

        self.space = CompositeSpace(
            (Conv2DSpace(shape=size,
                          num_channels=num_channels,
                          axes=('b', 0, 1, 'c')),
            VectorSpace(dim=(self.one_hot and 7 or 1))))
        self.source = ('features', 'targets')
        self.data_specs = (self.space, self.source)

        self.n_samples = len(self.y)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None, data_specs=None, return_tuple=True, rng=None):
	if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            raise ValueError('iteration mode not provided and no default '
                             'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        if data_specs is None:
            data_specs = getattr(self, '_iter_data_specs', None)

        # TODO: figure out where to to the scaling more cleanly.
        def list_to_scaled_array(batch):
            # batch is either a 4D ndarray, or a list of length 1
            # containing a 4D ndarray. Make it a 5D ndarray,
            # with shape 1 on the first dimension.
            # Also scale it from [0, 255] to [0, 1]
            if isinstance(batch, list):
                assert len(batch) == 1
                batch = batch[0]
            batch = batch.astype(config.floatX)
            batch /= 255.
            return batch[np.newaxis]

        convert_fns = []
        for space in data_specs[0].components:
            if (isinstance(space, Conv2DSpace) and
                    space.axes[0] == 'b'):
                convert_fns.append(list_to_scaled_array)
            else:
                convert_fns.append(None)

        return EmotiwFrameIterator(self,  mode(self.n_samples, batch_size, num_batches, rng),
					data_specs = data_specs, return_tuple=return_tuple,
					convert_fns=convert_fns)

    def get_data_specs(self):
        return self.data_specs
