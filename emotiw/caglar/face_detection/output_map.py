import tables
import os
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import SequentialSubsetIterator
import numpy

import theano
floatX = theano.config.floatX


class OutputMapFile(object):
    filters = tables.Filters(complib='blosc', complevel=1)
    """
        Size of the output map file.
    """
    def __init__(self, path):
        self.path = path

    @staticmethod
    def create_file(path, filename, n_examples, out_shape):
        assert filename is not None
        assert path is not None
        fileloc = os.path.join(path, filename)
        h5file = tables.openFile(fileloc, title="Output maps", mode="w")
        atom = tables.Float32Atom() if floatX == 'float32' else tables.Float64Atom()

        filters = OutputMapFile.filters
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")

        h5file.createCArray(gcolumns, 'Pt', atom = atom, shape = out_shape,
                                                title = "Output Image Patches", filters = filters)

        h5file.createCArray(gcolumns, 'Ino', atom = atom, shape = (n_examples,),
                                                title = "Image numbers", filters = filters)

        h5file.createCArray(gcolumns, 'Ploc', atom = atom, shape = (n_examples,),
                                                title = "Image locations", filters = filters)

        h5file.createCArray(gcolumns, 'Tgt', atom = atom, shape = (n_examples,),
                                                title = "Targets", filters = filters)

        return h5file, gcolumns

    @staticmethod
    def load_file(path, filename):
        fileloc = os.path.join(path, filename)
        h5file = tables.openFile(fileloc, title="Output maps", mode="r")
        return h5file

    @staticmethod
    def save_output_map(h5file, patches, imgnos, ilocs, targets, start=None, stop=None):
        if start and stop is not None:
            h5file.root.Data.Pt[start:stop] = patches
            h5file.root.Data.Ino[start:stop] = imgnos
            h5file.root.Data.Ploc[start:stop] = ilocs
            h5file.root.Data.Tgt[start:stop] = targets
        else:
            h5file.root.Data.Pt[:] = patches
            h5file.root.Data.Ino[:] = imgnos
            h5file.root.Data.Ploc[:] = ilocs
            h5file.root.Data.Tgt[:] = targets
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

    def extract_patches_batch(self,
            images,
            out_maps,
            start,
            conv_targets):
        """
        Get the patches that were able to survive in the convolutional training.
        """
        assert images.shape[0] == out_maps.shape[0]

        n_images = len(images)
        img_idx = start

        cropped_patches = []
        targets = []
        img_nos = []
        patch_locs = []

        for out_map in out_maps:
            non_empty_rfs = numpy.where(out_map!=0)
            for non_empty_rf in non_empty_rfs[0]:
                x = non_empty_rf % self.out_shape[1]
                y = int(non_empty_rf / self.out_shape[0])
                target = conv_targets[img_idx, non_empty_rf]
                image = numpy.reshape(images[img_idx], newshape=self.img_shape)
                patch = image[x: x + self.out_shape[0], y: y + self.out_shape[1]]
                cropped_patch = patch.flatten()
                cropped_patches.append(cropped_patch)
                patch_locs.append(non_empty_rf)
                targets.append(target)
                img_nos.append(img_idx)
            img_idx += 1
            assert n_images > img_idx

        cropped_patches = numpy.asarray(cropped_patches)
        targets = numpy.asarray(targets)
        patch_locs = numpy.asarray(patch_locs)
        return (cropped_patches, targets, patch_locs, img_nos)

    def get_next_patches(self,
            patches,
            out_map,
            img_nos,
            conv_targets):
        """
        ***NOTE***:
        This function is not being used and not very useful for our purposes.
        Probably requires a rewrite. See remove_nonfaces function in the
        multiscale_cascade_ensembles.py.
        Return the patches that are saved from the non_max_sup and thresholding.
        """
        cropped_patches = []
        targets = []
        img_nos_p = []
        patch_locs = []
        non_empty_rfs = numpy.where(out_map != 0)

        patch_count = 0
        rf_count = 0

        for patch in patches:
            if conv_targets[patch_count] != 0:
                non_empty_rf = non_empty_rfs[rf_count]
                x = non_empty_rf % self.out_shape[1]
                y = int(non_empty_rf / self.out_shape[0])

                target = conv_targets[patch_count]
                img_nos_p.append(patch)
                cropped_patches.append(patch)
                patch_locs.append(non_empty_rf)
                targets.append(conv_targets[non_empty_rf])
                rf_count +=1

            patch_count += 1

        cropped_patches = numpy.asarray(cropped_patches)
        targets = numpy.asarray(targets)
        patch_locs = numpy.asarray(patch_locs)
        img_nos_p = numpy.asarray(img_nos_p)
        return (cropped_patches, targets, patch_locs, img_nos_p)

class CroppedPatchesDataset(Dataset):
    """
    CroppedPatchesDataset is by itself an iterator.
    """
    def __init__(self,
                 img_shape,
                 iter_mode="fprop",
                 h5_file=None,
                 start=None,
                 stop=None,
                 mode=None):

        self.__dict__.update(locals())
        self.img_shape = img_shape

        if self.self is not None:
            del self.self

        if mode is not None:
            self.mode = mode
        elif start is not None or stop is not None:
            self.mode = "r+"
        else:
            self.mode = "r"

        if not os.path.isfile(h5_file):
            raise ValueError("Please enter a valid file path.")

        self.initialize_dataset(h5_file)

    def initialize_dataset(self, h5_file):
        """
        Set the files and the patches,...etc.
        """
        self.h5file = tables.openFile(h5_file, mode=self.mode)
        self.dataset = self.h5file.root

        self.X = self.dataset.Data.Pt
        self.Y = self.dataset.Data.Tgt
        self.imgnos = self.dataset.Data.Ino
        self.plocs = self.dataset.Data.Ploc
        self.data_n_rows = self.targets.shape[0]

    def set_iter_mode(self, r_mode):
        self.iter_mode = r_mode

    def get_design_matrix(self):
        """
        Return the patches as a dense design matrix.
        """
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
        self.cur_idx = -1

        if mode == 'sequential':
            self.subset_iterator = SequentialSubsetIterator(self.data_n_rows,
                                            batch_size=1, num_batches=num_batches, rng=None)
            return self
        else:
            raise NotImplementedError('other iteration scheme not supported for now!')

    def __iter__(self):
        return self

    def next(self):
        """
        Method for the getting the next indices from the minibatch.
        """

        if self.cur_idx == -1:
            batch_start_indx = self.subset_iterator.next()
        else:
            batch_start_indx = self.cur_idx

        begining_img_no = self.imgnos[batch_start_indx]

        mini_batch_patches = []
        mini_batch_plocs = []
        mini_batch_imgnos = []
        mini_batch_targets = []

        indx = batch_start_indx

        while indx is not None:
            if (mini_batch_targets[-1] is not None) and (mini_batch_imgnos[-1] != begining_img_no):
                self.cur_idx = indx
                break
            try:
                mini_batch_patches.append(self.X[indx.start])
                mini_batch_targets.append(self.Y[indx.start])
                mini_batch_imgnos.append(self.imgnos[indx.start])
                mini_batch_plocs.append(self.plocs[indx.start])
            except IndexError:
                print "The index of minibatch goes beyond the boundary."
                import ipdb; ipdb.set_trace()

            indx = self.subset_iterator.next()

        if self.iter_mode == "train":
            return (mini_batch_patches, mini_batch_targets)
        else:
            return (mini_batch_patches, mini_batch_targets, mini_batch_imgnos, mini_batch_plocs)

