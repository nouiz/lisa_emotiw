import numpy
import cv2

import warnings
import math

try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far MdMnist is "
            "only supported with PyTables")

from pylearn2.datasets.preprocessing import Preprocessor
from pylearn2.datasets import preprocessing
import os

class LaplacianPyramid(Preprocessor):
    """
    Create Laplacian pyramids for the multiscale architecture.

    Parameters:
    ----------
    nlevels: integer, obligatory
        This specifies the number of levels in the pyramid.

    subsampling_rates: list, optional
        The subsampling ratio to use at each level of the pyramid.

    img_shape: list, optional
        The shape of the original image.
    """
    def __init__(self,
            nlevels=None,
            subsampling_rates=None,
            img_shape=None, batch_size=1000,
            nchannels=1, paths=None,
            preprocess =False):

        if nchannels is not 1:
            raise ValueError("This class only supports the single channel images.")

        self.preprocess = preprocess
        assert nlevels is not None
        self.nlevels = nlevels
        self.img_shape = img_shape
        self.subsampling_rates = subsampling_rates
        self.paths = paths
        if self.paths is not None:
            assert len(paths) == nlevels

    def get_h5file(self, path, x_shape, y_shape):
        """
        Save to the specified path with the given shapes.
        """
        h5file, gcols = Imagenet.init_hdf5(path, (x_shape, y_shape))
        return h5file, gcols

    def get_h5files(self, data_size, pyr_shapes):
        """
            Save to the specified path with the given shapes.
        """
        h5files = []
        gcolumns = []

        assert self.paths is not None
        i = 0
        for path in self.paths:
            x_shape = (data_size, pyr_shapes[i][0]*pyr_shapes[i][1])
            y_shape = (data_size, )
            h5file, gcols = Imagenet.init_hdf5(path, (x_shape, y_shape))
            h5files.append(h5file)
            gcolumns.append(gcols)
            i += 1
        return h5files, gcolumns

    def get_h5files_processed(self, data_size, pyr_shapes):
        """
            Save to the specified path with the given shapes.
        """
        h5files = []
        gcolumns = []

        assert self.paths is not None
        i = 0
        for path in self.paths:
            path = os.path.splitext(path)[0] + 'preproced' + os.path.splitext(path)[-1]
            x_shape = (data_size, pyr_shapes[i][0]*pyr_shapes[i][1])
            y_shape = (data_size, )
            h5file, gcols = Imagenet.init_hdf5(path, (x_shape, y_shape))
            h5files.append(h5file)
            gcolumns.append(gcols)
            i += 1
        return h5files, gcolumns

    def get_pyr_shapes(self):
        pyr_shapes = [self.img_shape]
        for i in xrange(self.nlevels):
            if self.subsampling_rates is not None:
                img_shape = (pyr_shapes[-1][0] / self.subsampling_rates[i],
                        pyr_shapes[-1][1] / self.subsampling_rates[i])
            else:
                img_shape = (int(math.floor((pyr_shapes[-1][0] + 1) / 2)),
                        int(math.floor((pyr_shapes[-1][1] + 1) / 2)))
            pyr_shapes.append(img_shape)
        return pyr_shapes

    def gen_pyr_to_path(self, dataset, batch_size=5000):
        """
            Generate the laplacian pyramids with respect to given subsampling rates and number of
        levels.
            imgs: ndarray
                List of images.
            can_fit: bool
                Whether the datasets can fit the memory or not.
        """

        data_X = dataset
        #data_Y = dataset.y

        pyr_shapes = self.get_pyr_shapes()
        data_size = data_X.shape[0]

        prev_shape = self.img_shape
        h5files, gcols = self.get_h5files(data_size, pyr_shapes)

        if self.preprocess:
            GCN = preprocessing.GlobalContrastNormalizationPyTables()
            h5files_processed, gcols_processed = self.get_h5files_processed(data_size, pyr_shapes)


        for i in xrange(data_size):
            img = data_X[i]
            for l in xrange(self.nlevels):
                img = numpy.reshape(img, newshape=pyr_shapes[l])
                if self.subsampling_rates is not None:
                    next_img = cv2.pyrDown(img, dstsize=pyr_shapes[l+1])

                else:
                    next_img = cv2.pyrDown(img)

                tmp_img = cv2.pyrUp(next_img, dstsize=pyr_shapes[l])
                print 'level', l
                if self.preprocess:
                    temp = img - tmp_img
                    temp = GCN.transform(temp)
                    (w,h) = temp.shape
                    LCN = preprocessing.LeCunLCN((w, h), channels=[0], kernel_size=7)
                    temp = LCN.transform(temp.reshape((1,w,h,1)))
                    h5files_processed[l].root.Data.X[i] = temp.reshape((numpy.prod(pyr_shapes[l])))
                    h5files_processed[l].flush()

                l_img = numpy.reshape(img-tmp_img, newshape=numpy.prod(pyr_shapes[l]))

                h5files[l].root.Data.X[i] = l_img
                img = next_img
                h5files[l].flush()

        for h5file in h5files:
            h5file.close()
        if self.preprocess:
            for h5file in h5files_processed:
                h5file.close()


    def gen_pyr(self, dataset):
        """
            Generate the laplacian pyramids with respect to given subsampling rates and number of
        levels.

            imgs: ndarray
                List of images.
            can_fit: bool
                Whether the datasets can fit the memory or not.
        """
        imgs = dataset[0]
        bboxes = dataset[1]

        try:
            img_iterator = iter(imgs)
        except:
            raise ValueError(imgs, " is not iterable.")

        if imgs.ndim != 3:
            imgs = numpy.reshape(imgs, newshape=(self.img_shape[0], self.img_shape[1], -1))
        pyramids = []
        for img in img_iterator:
            prev_size = img.shape
            levels = []
            for i in xrange(self.nlevels-1):

                if self.subsampling_rates is not None:
                    size = (prev_size[0]/self.subsampling_rates[i],
                            prev_size[1]/self.subsampling_rates[i])
                    next_img = cv2.pyrDown(img, dstsize=size)
                else:
                    next_img = cv2.pyrDown(img)

                tmp_img = cv2.pyrUp(next_img, dstsize=prev_size)
                levels.append(img - tmp_img)
                img = next_img
                prev_size = img.shape

            #levels.append((img, bbox))
            pyramids.append(levels)
        pyramids = numpy.asarray(pyramids)
        return pyramids

    def apply(self, dataset, canfit=False):
        if self.paths is not None:
            self.gen_pyr_to_path(dataset)
        else:
            return self.gen_pyr(dataset)

