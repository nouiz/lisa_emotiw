import tables
import os
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import SequentialSubsetIterator
import numpy
import warnings

try:
        import scipy.sparse
except ImportError:
        warnings.warn("Couldn't import scipy.sparse")

import theano
import gzip
floatX = theano.config.floatX

class OutputMapFile(object):
    """
        Size of the output map file.
    """
    def __init__(self, path):
        self.path = path

    @staticmethod
    def create_file(path, filename, n_examples, out_shape):
        fileloc = os.path.join(path, filename)
        h5file = tables.openFile(fileloc, title="Output maps", "w")
        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()
        filters = DenseDesignMatrixPyTables.filters
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        h5file.createCArray(gcolumns, 'Pt', atom = atom, shape = out_shape,
                                                title = "Output Image Patches", filters = filters)

        h5file.createCArray(gcolumns, 'Ino', atom = atom, shape = (n_examples,),
                                                title = "Image numbers", filters = filters)

        h5file.createCArray(gcolumns, 'Iloc', atom = atom, shape = (n_examples,),
                                                title = "Image locations", filters = filters)

        h5file.createCArray(gcolumns, 'Tgt', atom = atom, shape = (n_examples,),
                                                title = "Targets", filters = filters)

        return h5file, gcolumns

    @staticmethod
    def load_file(path, filename, n_examples, out_shape):
        fileloc = os.path.join(path, filename)
        h5file = tables.openFile(fileloc, title="Output maps", "r")
        return h5file

    def save_output_map(self, h5file, output_map, imgnos, ilocs, targets, start=None, stop=None):

        if start and stop is not None:
            h5file.root.Data.Pt[start:stop] = output_map
            h5file.root.Data.Ino[start:stop] = imgnos
            h5file.root.Data.Iloc[start:stop] = ilocs
            h5file.root..Data.Tgt[start:stop] = targets
        else:
            h5file.root.Data.Pt = output_map
            h5file.root.Data.Ino[start:stop] = imgnos
            h5file.root.Data.Iloc[start:stop] = ilocs
            h5file.root.Data.Tgt[start:stop] = targets

        h5file.flush()

class OutputMap(object):
    """
    Parameters:
    ----------
    receptive_field_shape: Shape of the receptive field.
    img_shape: shape of the images.
    stride: Stride of the for the creation of the output maps.
    """
    def __init__(self, receptive_field_shape, img_shape, stride):
        self.receptive_field_shape = receptive_field_shape
        self.stride = stride
        self.img_shape = img_shape

        self.out_shape = [(receptive_field_shape[0] - img_shape[0]) / stride,
                (receptive_field_shape[1] - img_shape[1]) / stride]

    def extract_patches(self, images, out_maps):
        n_images = len(images)
        i = 0
        img_patches = []
        for out_map in out_maps:
            cropped_patches = []
            non_empty_rfs = numpy.where(out_maps!=1)
            for non_empty_rf in non_empty_rfs:
                x = non_empty_rfs % self.out_shape[1]
                y = int(non_empty_rfs / self.out_shape[0])
                image = numpy.reshape(images[i], newshape=self.img_shape)
                patch = image[x:x+self.out_shape[0], y:y+self.out_shape[1]]
                cropped_patch = patch.flatten()
                cropped_patches.append(cropped_patch)
            img_patches.append(cropped_patches)
        cropped_patches = numpy.asarray(img_patches)
        return cropped_patches

class CroppedPatchesDataset(Dataset):
    """
    CroppedPatchesDataset is by itself an iterator.
    """
    def __init__(self, img_shape, load_path=None, mode=None):

        self.load_path = load_path
        assert which_set in self.data_mapper.keys()
        self.__dict__.update(locals())
        del self.self

        if path is None:
            raise ValueError("The path variable should not be empty!")

        if mode is not None:
            mode = mode
        elif start != None or stop != None:
            mode = "r+"
        else:
            mode = "r"

        if not os.path.isfile(h5_file):
            raise ValueError("Please enter a valid file path.")

        self.h5file = tables.openFile(h5_file, mode=mode)
        dataset = self.h5file.root

        self.patches = self.h5file.root.Data.Pt
        self.targets = self.h5file.root.Data.Tgt
        self.imgnos = self.h5file.root.Data.Ino
        self.imglocs = self.h5file.root.Data.Iloc

    def get_design_matrix(self):
        return self.patches

    def get_batch_design(self, batch_size, include_labels=False):
        """
            Method inherited from the Dataset.
        """
        self.iterator(mode='sequential', batch_size=batch_size, num_batches=None, topo=None)
        return self.next()

    def get_batch_topo(self, batch_size):
        """
            Method inherited from the Dataset.
        """
        raise NotImplementedError('Not implemented for sparse dataset')

    def iterator(self,
            mode=None,
            batch_size=None,
            num_batches=None,
            topo=None,
            targets=None,
            rng=None):
        """
            Method inherited from the Dataset.
        """
        self.mode = mode
        self.batch_size = batch_size
        self._targets = targets

        if mode == 'sequential':
            self.subset_iterator = SequentialSubsetIterator(self.data_n_rows,
                                            batch_size, num_batches, rng=None)
            return self
        else:
            raise NotImplementedError('other iteration scheme not supported for now!')

    def __iter__(self):
        return self

    def next(self):
        indx = self.subset_iterator.next()
        try:
            mini_batch = self.sparse_matrix[indx]
        except IndexError:
            print "The index of minibatch goes beyond the boundary."
            import ipdb; ipdb.set_trace()
        return mini_batch
