from __future__ import division

import os
import warnings

import math
import numpy
from theano import config

try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far MdMnist is "
            "only supported with PyTables")

import functools
from pylearn2.utils.iteration import resolve_iterator_class

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

from pylearn2.space import CompositeSpace, Conv2DSpace

from pylearn2.utils import iteration
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess

from bbox import BoundingBox

from bbox_utils import convert_bboxes_exhaustive, convert_bboxes_guided, get_image_bboxes

class ConversionType:
    EXA = "Exhaustive"
    GUID = "Guided"

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
            receptive_field_shape=None,
            stride=1,
            use_output_map=True,
            bbox_conversion_type=ConversionType.GUID,
            topo=None,
            targets=None,
            area_ratio=0.9,
            data_specs=None,
            return_tuple=False):
        self.targets = targets
        self.topo = topo

        self.bbox_conversion_type = bbox_conversion_type
        self.img_shape = img_shape
        self.receptive_field_shape = receptive_field_shape
        self.stride = stride
        self.area_ratio = area_ratio
        self.use_output_map = use_output_map

        super(FaceBBoxDDMIterator, self).__init__(dataset,
                subset_iterator,
                topo=None,
                targets=None,
                data_specs=data_specs,
                return_tuple=return_tuple)

    def next(self):
        next_index = self._subset_iterator.next()
        if isinstance(next_index, numpy.ndarray) and len(next_index) == 1:
            next_index = next_index[0]
        self._needs_cast = True

        if self._needs_cast:
            features = numpy.cast[config.floatX](self._raw_data[0][next_index])
        else:
            features = self._raw_data[0][next_index,:]
        #import ipdb; ipdb.set_trace()
        if self.topo:
            if len(features.shape) != 2:
                features = features.reshape((1, features.shape[0]))
            features = self._dataset.get_topological_view(features)
        if self.targets:
            bbx_targets = get_image_bboxes(next_index, self._raw_data[1])
            if len(bbx_targets.shape) != 2:
                bbx_targets = bbx_targets.reshape((1, bbx_targets.shape[0]))

            if self.use_output_map:
                if self.bbox_conversion_type == ConversionType.GUID:
                    if isinstance(self._data_specs[0].components[1], Conv2DSpace):
                        conv_outs = True
                    else:
                        conv_outs = False
                    n_channels = self._data_specs[0].components[1].num_channels
                    targets = convert_bboxes_guided(bbx_targets, self.img_shape,
                            self.receptive_field_shape, self.area_ratio, self.stride,
                            conv_outs=conv_outs, n_channels=n_channels)
                else:
                    targets = convert_bboxes_exhaustive(bbx_targets, self.img_shape,
                            self.receptive_field_shape, self.area_ratio, self.stride)
            else:
                targets = self.get_bare_outputs(bbx_targets)

            if targets.shape[0] != features.shape[0]:
                raise ValueError("There is a batch size mismatch between features and targets.")

            self._targets_need_cast = True
            if self._targets_need_cast:
                targets = numpy.cast[config.floatX](targets)
            return features, targets
        else:
            if self._return_tuple:
                features = (features,)
            return features

    def get_bare_outputs(self, bbx_targets):
        targets = []
        for i in xrange(bbx_targets.shape[0]):
            bbox = bbx_targets[i]

            r = bbox.cols.row
            c = bbox.cols.col
            width = bbox.cols.width
            height = bbox.cols.height

            target = [r, c, width, height]
            targets.append(target)
        targets = numpy.asarray(target)
        return targets


class FaceBBoxDDMPytables(dense_design_matrix.DenseDesignMatrix):
    filters = tables.Filters(complib='blosc', complevel=1)
    h5file = None
    """
    DenseDesignMatrix based on PyTables for face bounding boxes.
    """
    def __init__(self, X=None, h5file=None, topo_view=None, y=None,
                 view_converter=None, axes = ('b', 0, 1, 'c'),
                 image_shape=None, receptive_field_shape=None,
                 bbox_conversion_type=ConversionType.GUID,
                 area_ratio=None,
                 stride=None, use_output_map=True, rng=None):
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
        self.bbox_conversion_type = bbox_conversion_type
        self.h5file = h5file
        self.area_ratio = area_ratio
        self._deprecated_interface = True
        FaceBBoxDDMPytables.filters = tables.Filters(complib='blosc', complevel=1)


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

        if self.h5file.isopen and (self.h5file.mode == "w" or self.h5file.mode == "r+"):
            self.fill_hdf5(h5file=self.h5file,
                data_x=X,
                start=start)
        else:
            raise ValueError("H5File is not open or not in the writable mode!")

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
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        # build data_specs from topo and targets if needed
        if topo is None:
            topo = getattr(self, '_iter_topo', False)

        if data_specs[0] is not None:
            if isinstance(data_specs[0], Conv2DSpace) or isinstance(data_specs[0].components[0],
                    Conv2DSpace):
                topo = True

        if topo:
            # self.iterator is called without a data_specs, and with
            # "topo=True", so we use the default topological space
            # stored in self.X_topo_space
            assert self.X_topo_space is not None
            X_space = self.X_topo_space
        else:
            X_space = self.X_space

        if targets is None:
            if "targets" in data_specs[1]:
                targets = True
            else:
                targets = False

        if data_specs is None:
            if targets:
                assert self.y is not None
                y_space = data_specs[0].components[1]
                space = CompositeSpace(components=(X_space, y_space))
                source = ('features', 'targets')
            else:
                space = X_space
                source = 'features'

            print space
            data_specs = (space, source)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
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
            data_specs = self._iter_data_specs

        return FaceBBoxDDMIterator(self,
                                    mode(self.X.shape[0], batch_size, num_batches, rng),
                                    img_shape=self.image_shape,
                                    receptive_field_shape=self.receptive_field_shape,
                                    stride=self.stride,
                                    bbox_conversion_type=self.bbox_conversion_type,
                                    topo=topo,
                                    targets=targets,
                                    area_ratio=self.area_ratio,
                                    use_output_map=self.use_output_map,
                                    data_specs=data_specs,
                                    return_tuple=return_tuple)

    @staticmethod
    def init_hdf5(path=None, shapes=None):
        """
        Initialize hdf5 file to be used as a dataset
        """
        assert shapes is not None

        x_shape, y_shape = shapes
        print "init_hdf5"

        # make pytables
        if path is None:
            if FaceBBoxDDMPytables.h5file is None:
                raise ValueError("path variable should not be empty.")
            else:
                h5file = FaceBBoxDDMPytables.h5file
        else:
                h5file = tables.openFile(path, mode = "w", title = "Google Face bounding boxes Dataset.")

        gcolumns = h5file.createGroup(h5file.root, "Data", "Data")
        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()

        filters = FaceBBoxDDMPytables.filters

        h5file.createCArray(gcolumns, 'X', atom = atom, shape = x_shape,
                title = "Images", filters = filters)

        h5file.createTable(gcolumns, 'bboxes', BoundingBox,
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
        if FaceBBoxDDMPytables.h5file is None:
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
    def resize(h5file, start, stop, remove_old_node=False):
        if h5file is None:
            raise ValueError("h5file should not be None.")

        data = h5file.root.Data
        node_name = "Data_%s_%s" % (start, stop)
        if remove_old_node:
            try:
                gcolumns = h5file.createGroup('/', node_name, "Data %s" %   node_name)
            except tables.exceptions.NodeError:
                h5file.removeNode('/', node_name, 1)
                gcolumns = h5file.createGroup('/', node_name, "Data %s" % node_name)
        elif node_name in h5file.root:
            return h5file, getattr(h5file.root, node_name)
        else:
            gcolumns = h5file.createGroup('/', node_name, "Data %s" %   node_name)

        if FaceBBoxDDMPytables.h5file is None:
            FaceBBoxDDMPytables.h5file = h5file

        start = 0 if start is None else start
        stop = gcolumns.X.nrows if stop is None else stop

        atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()
        filters = FaceBBoxDDMPytables.filters

        x = h5file.createCArray(gcolumns, 'X', atom = atom, shape = ((stop - start, data.X.shape[1])),
                title = "Images", filters = filters)

        y = h5file.createTable(gcolumns, 'bboxes', BoundingBox,
                title = "Face bounding boxes", filters = filters)

        x[:] = data.X[start:stop]
        bboxes = get_image_bboxes(slice(start, stop), data.bboxes)
        y.append(bboxes)

        if remove_old_node:
            h5file.removeNode('/', "Data", 1)
            h5file.renameNode('/', "Data", node_name)

        h5file.flush()
        return h5file, gcolumns

class FaceBBox(FaceBBoxDDMPytables):
    """
        This is the pylearn2 interface class for the Google Faces bounding
    boxes dataset.
    """

    data_mapper = {
            "train": 0,
            "valid": 1,
            "test": 2
    }

    def __init__(self,
            which_set,
            area_ratio,
            path=None,
            scale=False,
            center=False,
            start=None,
            stop=None,
            img_shape=None,
            mode=None,
            axes=('b', 0, 1, 'c'),
            bbox_conversion_type=ConversionType.GUID,
            use_output_map=False,
            size_of_receptive_field=None,
            stride=1,
            preprocessor=None):
        """
        m: The mode to open the h5file.
        bbox_conversion_type: What type of conversion to perform on the bounding boxes.
        There are two viable options:
            GUID: Perform Guided search.(faster)
            EXHAUSTIVE: Perform exhaustive search on the whole image.
        use_output_map: Whether to use the convolutional output maps or
        size_of_receptive: Size of the receptive field for the convolutional output map.
        area_ratio: In order to create an output map what ratio of the face should be in the
        receptive to be able to say that the receptive field contains a face.
        """

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

        path = preprocess(path)

        if path.endswith(".h5"):
            h5_file = path
        else:
            raise ValueError("This class only supports the exact file directories for the path constructor variable.")

        if not os.path.isfile(h5_file):
            raise ValueError("Please enter a valid file path.")

        self.h5file = tables.openFile(h5_file, mode=mode)
        dataset = self.h5file.root
        self.area_ratio = area_ratio

        if start != None or stop != None:
            self.h5file, data = self.resize(self.h5file, start, stop)

        images = data.X
        bboxes = data.bboxes

        if img_shape is None:
            img_shape = (256, 256)

        self.img_shape = img_shape

        if center or scale:
            raise ValueError("We don't support centering or scaling yet.")

        view_converter = dense_design_matrix.DefaultViewConverter((img_shape[0], img_shape[1], 1),
                axes)
        self._deprecated_interface = True

        super(FaceBBox, self).__init__(X=images,
                                    area_ratio=area_ratio,
                                    y=bboxes,
                                    h5file=self.h5file,
                                    image_shape=img_shape,
                                    receptive_field_shape=size_of_receptive_field,
                                    stride=stride,
                                    bbox_conversion_type=bbox_conversion_type,
                                    use_output_map=use_output_map,
                                    view_converter=view_converter)

        if preprocessor:
            can_fit = True
            preprocessor.apply(self, can_fit)

        if self.h5file is not None:
            self.h5file.flush()

