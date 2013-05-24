import os
import gc
import warnings

try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far MdMnist is "
            "only supported with PyTables")

import functools

import numpy
from theano import config
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

from pylearn2.utils import iteration
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess


class FaceBBoxDDMIterator(iteration.FiniteDatasetIterator):
    """
    Iterator class for Face bounding boxes dense desing matrix class.


    Parameters
    ----------

        dataset: DenseDesignMatrix
        img_shape: list
        receptive_field_shape: list
        stride: int
        subset_iterator: iterator
        topo: bool, optional
        targets: bool, optional
        area_ratio: float, optional
    """
    def __init__(self, dataset,
            subset_iterator,
            img_shape=None,
            receptive_field_shape=None, stride=1,
            use_output_map=True,
            topo=False, targets=False,
            area_ratio=0.9):

        self.img_shape = img_shape
        self.receptive_field_shape = receptive_field_shape
        self.stride = stride
        self.area_ratio = area_ratio
        self.use_output_map = use_output_map

        super(FaceBBoxDDMIterator, self).__init__(dataset, subset_iterator, topo=topo,
                targets=targets)

    def next(self):
        next_index = self._subset_iterator.next()
        if isinstance(next_index, numpy.ndarray) and len(next_index) == 1:
            next_index = next_index[0]
        if self._needs_cast:
            features = numpy.cast[config.floatX](self._raw_data[next_index])
        else:
            features = self._raw_data[next_index,:]
        if self._topo:
            if len(features.shape) != 2:
                features = features.reshape((1, features.shape[0]))
            features = self._dataset.get_topological_view(features)
        if self._targets:
            bbx_targets = self.get_image_bboxes(next_index)
            if len(bbx_targets.shape) != 2:
                bbx_targets = bbx_targets.reshape((1, bbx_targets.shape[0]))
            if self.use_output_map:
                targets = self.convert_bboxes(bbx_targets, self.img_shape, self.receptive_field_shape, self.stride)
            else:
                targets = self.get_bare_outputs(bbx_targets)

            if self._targets_need_cast:
                targets = numpy.cast[config.floatX](targets)
            return features, targets
        else:
            return features

    def get_image_bboxes(self, image_index):
        query = "imgno={}".format(image_index)
        return self._raw_targets.readWhere(query)

    def get_bare_outputs(self, bbx_targets):
        targets = []
        for i in xrange(bbx_targets.shape[0]):
            bbox = bbx_targets[i]
            r = bbox.field("row")
            c = bbox.field("col")
            width = bbox.field("width")
            height = bbox.field("height")
            target = [r, c, width, height]
            targets.append(target)
        targets = numpy.asarray(target)
        return targets

    def convert_bboxes(self, bbx_targets, img_shape, rf_shape, stride=1):
        """
        This function converts the bounding boxes to the spatial outputs for the neural network.
        In order to do this, we do a naive convolution and check if the bounding box is inside a
        given receptive field.
        Parameters
        ---------

        bbx_targets: pytables table.
        img_shape: list
        rf_shape: list
        stride: integer
        """
        assert bbx_targets is not None
        assert img_shape is not None
        assert rf_shape is not None

        output_maps = []
        for i in xrange(bbx_targets.shape[0]):
            output_map = []
            rf_y_start = 0
            rf_y_end = rf_shape[0]

            bbox = bbx_targets[i]
            r = bbox.field("row")
            c = bbox.field("col")
            width = bbox.field("width")
            height = bbox.field("height")

            #Perform convolution for each bounding box
            while (rf_y_end <= img_shape[0]):
                rf_x_start = 0
                rf_x_end = rf_shape[1]
                while (rf_x_end <= img_shape[1]):
                    #Check if any corner of the image falls inside the boundary box:
                    if perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r, c]):
                        x2 = min(rf_x_start + rf_shape[1], c + width)
                        y2 = min(rf_y_start + rf_shape[0], r + height)
                        s_w = x2 - c
                        s_h = y2 - r

                    elif perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r + height, c]):
                        x2 = min(rf_x_start + rf_shape[1], c + width)
                        y2 = r + height
                        s_w = x2 - c
                        s_h = y2 - rf_y_start

                    elif perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r,c+width]):
                        x2 = c + width
                        y2 = min(rf_y_start + rf_shape[0], r + height)
                        s_w = x2 - rf_x_start
                        s_h = y2 - rf_y_start

                    elif perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r+height,c+width]):
                        x2 = c + width
                        y2 = r + height
                        s_w = x2 - rf_x_start
                        s_h = y2 - rf_y_start

                    s_area = s_w * s_h
                    area = width * height

                    #If the face area is very small ignore it.
                    if area <= 18 or s_area <= 18:
                        ratio = 0.
                    else:
                        ratio = float(s_area) / float(area)

                    if ratio >= self.area_ratio:
                        output_map.append(1)
                    else:
                        output_map.append(0)

                    rf_x_start += stride
                    rf_x_end = rf_x_start + rf_shape[1]
                rf_y_start += stride
                rf_y_end = rf_y_start + rf_shape[0]
        output_maps = numpy.asarray(output_maps)
        return output_maps

def perform_hit_test(bbx_start, w, h, point):
    """
    Check if a point is in the bounding box.
    """
    if (bbx_start[0] <= point[0] and bbx_start[0] + w >= point[0]
            and bbx_start[1] +h >= point[1] and bbx_start <= point[1]):
        return True
    else:
        return False

class FaceBBoxDDMPytables(dense_design_matrix.DenseDesignMatrix):
    filters = tables.Filters(complib='blosc', complevel=1, fletcher32=True)

    """
    DenseDesignMatrix based on PyTables for face bounding boxes.
    """
    def __init__(self, X=None, topo_view=None, y=None,
                 view_converter=None, axes = ('b', 0, 1, 'c'),
                 image_shape=None, receptive_field_shape=None,
                 stride=None, use_output_map=False, rng=None):
        """
        Parameters
        ----------

        X : ndarray, 2-dimensional, optional
            Should be supplied if `topo_view` is not. A design
            matrix of shape (number examples, number features)
            that defines the dataset.
        topo_view : ndarray, optional
            Should be supplied if X is not.  An array whose first
            dimension is of length number examples. The remaining
            dimensions are xamples with topological significance,
            e.g. for images the remaining axes are rows, columns,
            and channels.
        y : ndarray, 1-dimensional(?), optional
            Labels or targets for each example. The semantics here
            are not quite nailed down for this yet.
        view_converter : object, optional
            An object for converting between design matrices and
            topological views. Currently DefaultViewConverter is
            the only type available but later we may want to add
            one that uses the retina encoding that the U of T group
            uses.
        image_shape: list
            Shape of the images that we are processing.
        receptive_field_size: list
            Size of the receptive field of the convolutional neural network.
        stride: integer
            The stride that we have used for the convolution operation.
        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
        """

        if rng is None:
            rng = (17, 2, 946)

        assert image_shape is not None
        assert receptive_field_shape is not None
        assert stride is not None

        self.image_shape = image_shape
        self.receptive_field_shape = receptive_field_shape
        self.stride = stride
        self.use_output_map = use_output_map

        self.h5file = None
        FaceBBoxDDMPytables.filters = tables.Filters(complib='blosc', complevel=1, fletcher32=True)


        super(FaceBBoxDDMPytables, self).__init__(X = X,
                                            topo_view = topo_view,
                                            y = y,
                                            view_converter = view_converter,
                                            axes = axes,
                                            rng = rng)
    def set_design_matrix(self, X, start = 0):
        """
        Parameters
        ----------
        X: Images
        """
        assert (len(X.shape) == 2)
        assert self.h5file is not None
        assert not numpy.any(numpy.isnan(X))
        self.fill_hdf5(h5file=self.h5file,
                data_x=X,
                start=start)

    def set_topological_view(self, V, axes=('b', 0, 1, 'c'), start=0):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
        TODO: why is this parameter named 'V'?
        """
        assert not numpy.any(numpy.isnan(V))
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
        X = self.view_converter.topo_view_to_design_mat(V)
        assert not numpy.any(numpy.isnan(X))
        FaceBBoxDDMPytables.fill_hdf5(h5file = self.h5file,
                                            data_x = X,
                                            start = start)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None,
            num_batches=None,
            topo=None, targets=None, rng=None):
        # TODO: Refactor, deduplicate with DenseDesignMatrix.iterator
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default'
                        'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)

        if topo is None:
            topo = getattr(self, '_iter_topo', False)

        if targets is None:
            targets = getattr(self, '_iter_targets', False)

        if rng is None and mode.stochastic:
            rng = self.rng

        return FaceBBoxDDMIterator(self,
                                    mode(self.X.shape[0], batch_size, num_batches, rng),
                                    self.image_shape,
                                    self.receptive_field_shape,
                                    self.stride,
                                    topo, targets,
                                    use_output_map=self.use_output_map)

    @staticmethod
    def init_hdf5(path, shapes):
        """
        Initialize hdf5 file to be used as a dataset
        """
        x_shape, y_shape = shapes

        # make pytables
        h5file = tables.openFile(path, mode = "w", title = "Google Face bounding boxes Dataset.")
        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()

        filters = FaceBBoxDDMPytables.filters

        h5file.createCArray(gcolumns, 'X', atom = atom, shape = x_shape,
                title = "Images", filters = filters)

        h5file.createCArray(gcolumns, 'bboxes', atom = atom, shape = y_shape,
                title = "Face bounding boxes", filters = filters)

        return h5file, gcolumns

    @staticmethod
    def fill_hdf5(h5file, data_x, data_y = None, node = None, start = 0, batch_size = 5000):
        """
        PyTables tends to crash if you write large data on them at once.
        This function write data on file in batches

        start: the start index to write data
        """

        if node is None:
            node = h5file.root.Data
        FaceBBoxDDMPytables.h5file = h5file
        data_size = data_x.shape[0]
        last = numpy.floor(data_size / float(batch_size)) * batch_size
        for i in xrange(0, data_size, batch_size):
            stop = i + numpy.mod(data_size, batch_size) if i >= last else i + batch_size
            assert len(range(start + i, start + stop)) == len(range(i, stop))
            assert (start + stop) <= (node.X.shape[0])
            node.X[start + i: start + stop, :] = data_x[i:stop, :]

            if data_y is not None:
                node.y[start + i: start + stop, :] = data_y[i:stop, :]

            h5file.flush()

    @staticmethod
    def resize(h5file, start, stop):
        data = h5file.root.Data
        try:
            gcolumns = h5file.createGroup('/', "Data_", "Data")
        except tables.exceptions.NodeError:
            h5file.removeNode('/', "Data_", 1)
            gcolumns = h5file.createGroup('/', "Data_", "Data")
        FaceBBoxDDMPytables.h5file = h5file
        start = 0 if start is None else start
        stop = gcolumns.X.nrows if stop is None else stop

        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()
        filters = FaceBBoxDDMPytables.filters
        x = h5file.createCArray(gcolumns, 'X', atom = atom, shape = ((stop - start, data.X.shape[1])),
                title = "Images", filters = filters)

        y = h5file.createCArray(gcolumns, 'bboxes', atom = atom, shape = ((stop - start, data.bboxes.shape[1])),
                title = "Bounding boxes", filters = filters)

        x[:] = data.X[start:stop]
        y[:] = data.bboxes[start:stop]

        h5file.removeNode('/', "Data", 1)
        h5file.renameNode('/', "Data", "Data_")
        h5file.flush()
        return h5file, gcolumns

class FaceBBox(FaceBBoxDDMPytables):
    data_mapper = {
            "train": 0,
            "valid": 1,
            "test": 2
    }
    def __init__(self,
            which_set, path=None, scale=False, center=False,
            start=None, stop=None, img_shape=None,
            use_output_map=False, size_of_receptive_field=None,
            stride=1, preprocessor=None):

        assert which_set in self.data_mapper.keys()
        self.__dict__.update(locals())
        del self.self

        if path is None:
            raise ValueError("The path variable should not be empty!")

        mode = "r+"
        path = preprocess(path)

        if path.endswith(".h5"):
            h5_file = path
        else:
            raise ValueError("This class only supports the exact file directories for the path constructor variable.")

        self.h5file = tables.openFile(h5_file, mode=mode)
        dataset = self.h5file.root
        if not os.path.isfile(h5_file):
            raise ValueError("Please enter a valid file path.")

        if start != None or stop != None:
            self.h5file, data = self.resize(self.h5file, start, stop)

        images = self.h5file.root.Data.X
        bboxes = self.h5file.root.Data.bboxes

        if img_shape is None:
            img_shape = (256, 256)
        self.img_shape = img_shape

        if center or scale:
            raise ValueError("We don't support centering or scaling yet.")
        view_converter = dense_design_matrix.DefaultViewConverter((img_shape[0], img_shape[1], 1))
        super(FaceBBox, self).__init__(X=images,
                                    y=bboxes,
                                    use_out_map=use_output_map,
                                    view_converter=view_converter)

        if preprocessor:
            can_fit = True
            preprocessor.apply(self, can_fit)
        self.h5file.flush()
