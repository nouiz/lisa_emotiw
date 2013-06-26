import numpy
import warnings

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import resolve_iterator_class
from bbox_utils import convert_bboxes_guided, get_image_bboxes

import copy


class DataArrangerIter(object):
    """
    Iterator for the data arranger class
    """
    _default_seed = (17, 2, 946)
    def __init__(self,
            inst,
            iter_mode=None,
            operate_on_patches=False,
            batch_size=None,
            topo=None,
            targets=True,
            rng=_default_seed):

        self.inst = inst
        self.face_ratio = inst.face_ratio
        self.operate_on_patches = operate_on_patches
        self.n_face_per_batch = int(batch_size * self.face_ratio)
        self.n_nonface_per_batch = int(batch_size * self.face_ratio)
        self.batch_size = batch_size

        self.face_dataset_iter = inst.face_dataset.iterator(mode=iter_mode,
                batch_size=self.n_face_per_batch,
                topo=topo, targets=targets)

        self.nonface_dataset_iter = inst.nonface_dataset.iterator(mode=iter_mode,
                batch_size=self.n_nonface_per_batch,
                topo=topo, targets=targets)

        self.rng = numpy.random.RandomState(rng)

        self.total_n_exs = self.inst.total_n_exs
        self.topo = topo
        self.targets = targets
        self.iter_mode = self.inst.subset_iterator

        self.face_img_idx = 0
        self.nonface_img_idx = 0

    @staticmethod
    def fisher_yates_shuffle(imgs, tgts):
        length = imgs.shape[0]
        for i in xrange(length):
            j = numpy.random.randint(0, length)
            imgs[i], imgs[j] = imgs[j], imgs[i]
            tgts[i], tgts[j] = tgts[j], tgts[i]
        return imgs, tgts

    @staticmethod
    def _mix_faces(non_face_imgs, face_imgs, non_face_tgts, face_tgts):
        face_imgs = numpy.concatenate((non_face_imgs, face_imgs))
        targets = numpy.concatenate((non_face_tgts, face_tgts))
        DataArrangerIter.fisher_yates_shuffle(face_imgs, face_tgts)
        return face_imgs, targets

    def next(self):

        images, targets, imgnos, imglocs, is_faces = [], [], [], [], []
        next_index = self.iter_mode.next()
        face_batch_idx = 0
        nonface_batch_idx = 0
        nonface_vals = self.nonface_dataset_iter.next()
        face_vals = self.face_dataset_iter.next()
        face_flag = False
        die_values = self.rng.rand(self.batch_size)

        for i in xrange(next_index.start, next_index.stop):
            die_value = die_values[face_batch_idx + nonface_batch_idx]
            if die_value >= self.face_ratio:
                if nonface_batch_idx < self.n_nonface_per_batch:
                    if not self.operate_on_patches:
                        self.inst.face_table.insert_face_table(imgno=self.nonface_img_idx, is_face=False)
                    face_flag = False
                    batch_idx = nonface_batch_idx
                    vals = nonface_vals
                else:
                    if not self.operate_on_patches:
                        self.inst.face_table.insert_face_table(imgno=self.face_img_idx, is_face=True)
                    face_flag = True
                    batch_idx = face_batch_idx
                    vals = face_vals
            else:
                if face_batch_idx < self.n_face_per_batch:
                    if not self.operate_on_patches:
                        self.inst.face_table.insert_face_table(imgno=self.face_img_idx, is_face=True)
                    face_flag = True
                    batch_idx = face_batch_idx
                    vals = face_vals
                else:
                    if not self.operate_on_patches:
                        self.inst.face_table.insert_face_table(imgno=self.nonface_img_idx, is_face=False)
                    face_flag = False
                    batch_idx = nonface_batch_idx
                    vals = nonface_vals

            if self.operate_on_patches:
                images.append(vals[0][batch_idx])
                targets.append(vals[1][batch_idx])
                imgnos.append(vals[2][batch_idx])
                imglocs.append(vals[3][batch_idx])
                is_faces.append(face_flag)
            else:
                images.append(vals[0][batch_idx])
                targets.append(vals[1][batch_idx])

                img_no = self.face_img_idx if face_flag else self.nonface_img_idx

                imgnos.append(img_no)
                is_faces.append(face_flag)

            if face_flag:
                self.face_img_idx += 1
                face_batch_idx += 1
            else:
                self.nonface_img_idx += 1
                nonface_batch_idx += 1

        if self.operate_on_patches:
            images = numpy.asarray(images)
            targets = numpy.asarray(targets)
            imgnos = numpy.asarray(imgnos)
            imglocs = numpy.asarray(imglocs)
            is_faces = numpy.asarray(is_faces)
            return images, targets, imgnos, imglocs
        else:
            images = numpy.asarray(images)
            targets = numpy.asarray(targets)
            is_faces = numpy.asarray(is_faces)
            return images, targets, imgnos

    def __iter__(self):
        return self


class DataArranger(Dataset):
    """
    This dataset interface takes two dataset object, preferably a dense design matrix and
    combines them online.
    """
    def __init__(self,
            face_dataset,
            nonface_dataset,
            face_ratio,
            operate_on_patches,
            face_table,
            total_n_exs):

        self.face_dataset = face_dataset
        self.nonface_dataset = nonface_dataset
        self.face_ratio = face_ratio
        self.total_n_exs = total_n_exs
        self.face_table = face_table
        self.is_first_cascade = False
        self.operate_on_patches = operate_on_patches

    def get_design_matrix(self, face_ddm=True):
        """
        TODO: Return a more generic dense design matrix.
        """
        if face_ddm:
            warnings.warn("By default, returning the face dataset as a dense design matrix. Might cause unexpected results.")
            return self.face_dataset.X
        else:
            return self.nonface_dataset.X

    def get_batch_design(self,
            batch_size,
            include_labels=False):
        """
        Method inherited from the Dataset.
        """
        iterator = self.iterator(mode='sequential',
                batch_size=batch_size,
                num_batches=None,
                topo=None)
        return iterator.next()

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
        mode = resolve_iterator_class(mode)

        self.subset_iterator = mode(self.total_n_exs,
                batch_size, num_batches, rng=None)

        return DataArrangerIter(self,
                mode,
                batch_size=batch_size,
                operate_on_patches=self.operate_on_patches)

    def get_face_data(self, imgno):
        return self.face_dataset.next()


class ExtractGridPatches(object):
    """
    Converts a dataset of images into a dataset of patches extracted along
    a regular grid from each image. The order of the images is preserved.
    """


    def __init__(self,
                patch_shape,
                area_ratio,
                patch_stride):

        self.area_ratio = area_ratio
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride

    def apply(self,
                dataset,
                start=None,
                stop=None,
                is_face=True,
                can_fit=False):
        """
        is_face: Bool,
            Flag that determines if we operate on face or nonface dataset.
        """

        if is_face:
            X = dataset.face_dataset.get_topological_view()
        else:
            X = dataset.nonface_dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractGridPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on"
                             + " dataset with " +
                             str(num_topological_dimensions) + ".")

        num_patches = X.shape[0]
        max_strides = [X.shape[0] - 1]
        for i in xrange(num_topological_dimensions):
            patch_width = self.patch_shape[i]
            data_width = X.shape[i + 1]
            last_valid_coord = data_width - patch_width
            if last_valid_coord < 0:
                raise ValueError('On topological dimension ' + str(i) +
                                 ', the data has width ' + str(data_width) +
                                 ' but the requested patch width is ' +
                                 str(patch_width))

            stride = self.patch_stride[i]
            if stride == 0:
                max_stride_this_axis = 0
            else:
                max_stride_this_axis = last_valid_coord / stride
            num_strides_this_axis = max_stride_this_axis + 1
            max_strides.append(max_stride_this_axis)
            num_patches *= num_strides_this_axis

        # batch size
        output_shape = [num_patches]

        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)

        # number of channels
        output_shape.append(X.shape[-1])
        output = numpy.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        coords = [0] * (num_topological_dimensions + 1)
        keep_going = True
        i = 0

        bboxes = dataset.face_dataset.bboxes

        while keep_going:

            args = [coords[0]]

            for j in xrange(num_topological_dimensions):
                coord = coords[j + 1] * self.patch_stride[j]
                args.append(slice(coord, coord + self.patch_shape[j]))

            args.append(channel_slice)
            patch = X[args]
            output[i, :] = patch
            i += 1

            # increment coordinates
            j = 0
            keep_going = False

            while not keep_going:
                if coords[-(j + 1)] < max_strides[-(j + 1)]:
                    coords[-(j + 1)] += 1
                    keep_going = True
                else:
                    coords[-(j + 1)] = 0
                    if j == num_topological_dimensions:
                        break
                    j = j + 1

        portion = slice(start, stop)

        repeat_times = numpy.ones(X.shape[0], 1)
        if is_face:
            bbox_targets = get_image_bboxes(portion, bboxes)

            outputmaps = convert_bboxes_guided(bbox_targets, (X.shape[0], X.shape[1]),
                self.patch_shape,
                area_ratio=self.area_ratio,
                stride=stride)
        else:
            targets = numpy.zeros(output_shape[-1])
            outputmaps = repeat_times * targets

        patch_loc = numpy.arange(output_shape[-1])
        patch_locs = repeat_times * patch_loc

        if start is None and stop is None:
            return output, outputmaps.flatten(), patch_locs.flatten()
        elif start is None:
            raise ValueError("You should give a start value.")
        elif start is not None and stop is None:
            return output[start], outputmaps.flatten(), patch_locs[start].flatten()
        else:
            return output[start:stop], outputmaps.flatten(), patch_locs[start:stop].flatten()


class ExtractRandomPatches(object):
    """
    Converts an image dataset into a dataset of patches extracted at random
    from the original dataset.
    """
    def __init__(self,
         patch_shape,
         num_patches,
         rng=None):

        self.patch_shape = patch_shape
        self.num_patches = num_patches
        if rng is not None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = numpy.random.RandomState([1, 2, 3])

    def apply(self,
                dataset,
                start=None,
                stop=None,
                can_fit=False):

        rng = copy.copy(self.start_rng)

        X = dataset.nonface_dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # batch size
        output_shape = [self.num_patches]

        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)

        # number of channels
        output_shape.append(X.shape[-1])
        output = numpy.zeros(output_shape, dtype=X.dtype)

        channel_slice = slice(0, X.shape[-1])
        targets = numpy.zeros(output_shape[-1])

        for i in xrange(self.num_patches):
            args = []
            args.append(rng.randint(X.shape[0]))

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                args.append(slice(coord, coord + self.patch_shape[j]))

            args.append(channel_slice)
            output[i, :] = X[args]

        patch_loc = numpy.arange(output_shape[-1])
        repeat_times = numpy.ones(X.shape[0], 1)
        patch_locs = repeat_times * patch_loc

        if start is None and stop is None:
            return output, targets, patch_locs
        elif start is None:
            raise ValueError("You should give a start value.")
        elif start is not None and stop is None:
            return output[start], targets[start].flatten(), patch_locs[start].flatten()
        else:
            return (output[start:stop], targets[start:stop].flatten(),
                patch_locs[start:stop].flatten())

