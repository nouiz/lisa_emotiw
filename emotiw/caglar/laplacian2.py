import numpy
import cv2

import warnings

try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far MdMnist is "
            "only supported with PyTables")

from pylearn2.datasets.preprocessing import Preprocessor
from face_bbox import FaceBBoxDDMPytables

"""
Get the scaled bounding boxes.
"""
def get_scaled_bbox(bbox, orig_size, scaled_size):
    x_scale_ratio = float(scaled_size[0]) / float(orig_size[0])
    y_scale_ratio = float(scaled_size[1]) / float(orig_size[1])
    bbox[2] = int(bbox[2] * y_scale_ratio)
    bbox[3] = int(bbox[3] * x_scale_ratio)
    bbox[4] = int(bbox[4] * y_scale_ratio)
    bbox[5] = int(bbox[5] * x_scale_ratio)
    return bbox

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
            iterator,
            nlevels=None,
            subsampling_rates=None,
            img_shape=None, batch_size=1000,
            nchannels=1, paths=None):

        if nchannels is not 1:
            raise ValueError("This class only supports the single channel images.")

        assert nlevels is not None
        self.nlevels = nlevels
        self.img_shape = img_shape
        self.subsampling_rates = subsampling_rates
        self.paths = paths
        if self.paths is not None:
            assert len(paths) == nlevels

    def get_h5files(self, x_shape, y_shape):
        """
            Save to the specified path with the given shapes.
        """
        h5files = []
        gcolumns = []

        assert self.paths is not None

        for path in self.paths:
            h5file, gcols = FaceBBoxDDMPytables.init_hdf5(path, (x_shape, y_shape))
            h5files.append(h5file)
            gcolumns.append(gcols)

        return h5files, gcolumns

    def get_h5file(self, path, x_shape, y_shape):
        """
            Save to the specified path with the given shapes.
        """
        h5file, gcols = FaceBBoxDDMPytables.init_hdf5(path, (x_shape, y_shape))
        return h5file, gcols

    def gen_pyr_to_path(self, dataset, batch_size=100):
        """
            Generate the laplacian pyramids with respect to given subsampling rates and number of
        levels.
            imgs: ndarray
                List of images.
            can_fit: bool
                Whether the datasets can fit the memory or not.
        """

        data_X = dataset.X
        data_Y = dataset.y

        prev_shape = self.img_shape
        h5files = []

        import math

        for l in xrange(self.nlevels):
            print "Creating level %d" % l
            x_shp, y_shp = data_X.shape, data_Y.shape

            if self.subsampling_rates is not None:
                img_shape = (self.img_shape[0] / numpy.prod(self.subsampling_rates[0:l]),
                        self.img_shape[1] / numpy.prod(self.subsampling_rates[0:l]))
            else:
                img_shape = (int(math.floor((self.img_shape[0]+1)/(2**(l+1)))),
                        int(math.floor((self.img_shape[1]+1)/(2**(l+1)))))

            h5file, gcol = self.get_h5file(self.paths[l], (x_shp[0], prev_shape[0]*prev_shape[1]),
                    y_shp)
            h5files.append(h5file)
            data_size = x_shp[0]

            last = numpy.floor(data_size / float(batch_size)) * batch_size

            images = h5files[l].root.Data.X
            table = h5files[l].root.Data.bboxes


            for i in xrange(0, data_size, batch_size):
                start = i
                stop = i + batch_size if i < last else i + numpy.mod(data_size, batch_size)
                data_X_batches = data_X[start:stop]
                data_y_batches = data_Y[start:stop]

                data_X_lst = []
                tbl_bboxes = table.row
                for in xrange(self.nlevels):
                    for j in xrange(start, stop):
                        newshape = (prev_shape[0], prev_shape[1])
                        img = numpy.reshape(data_X_batches[j], newshape=newshape)
                        resized_bbox = get_scaled_bbox(data_y_batches[j], prev_shape, img_shape)
                        tbl_bboxes["picasaBatchNumber"] = resized_bbox[0]
                        tbl_bboxes["idxInPicasaBatch"] = resized_bbox[1]
                        tbl_bboxes["faceno"] = data_y_batches[j]["faceno"]
                        tbl_bboxes["imgno"] = data_y_batches[j]["imgno"]
                        tbl_bboxes["row"] = resized_bbox[2]
                        tbl_bboxes["col"] = resized_bbox[3]
                        tbl_bboxes["height"] = resized_bbox[4]
                        tbl_bboxes["width"] = resized_bbox[5]
                        tbl_bboxes.append()

                        if self.subsampling_rates is not None:
                            next_img = cv2.pyrDown(img, dstsize=img_shape)
                        else:
                            next_img = cv2.pyrDown(img)

                        tmp_img = cv2.pyrUp(next_img, dstsize=(prev_shape[0], prev_shape[1]))
                        data_X_lst.append(numpy.reshape(img - tmp_img, newshape=(prev_shape[0]*prev_shape[1])))

                    table.flush()
                    images[start:stop] = numpy.asarray(data_X_lst)

            prev_shape = img_shape
            data_X = images
            data_Y = h5files[l].root.Data.bboxes
            prev_shape = img_shape

        for h5file in h5files:
            h5file.flush()

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

