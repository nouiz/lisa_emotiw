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

def get_image_bboxes(image_idx,  bboxes):
    """
        Query pytables table for the given range of images.
    """
    start = image_idx
    stop = image_idx
    query = "(imgno>={}) & (imgno<={})".format(start, stop)
    return bboxes.readWhere(query)


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

    def get_h5file(self, path, x_shape, y_shape):
        """
            Save to the specified path with the given shapes.
        """
        h5file, gcols = FaceBBoxDDMPytables.init_hdf5(path, (x_shape, y_shape))
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
            h5file, gcols = FaceBBoxDDMPytables.init_hdf5(path, (x_shape, y_shape))
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

        data_X = dataset.X
        data_Y = dataset.y

        pyr_shapes = self.get_pyr_shapes()
        data_size = data_X.shape[0]

        prev_shape = self.img_shape
        h5files, gcols = self.get_h5files(data_size, pyr_shapes)

        for i in xrange(data_size):
            img = data_X[i]
            bboxes = get_image_bboxes(i, data_Y)

            for l in xrange(self.nlevels):
                table = h5files[l].root.Data.bboxes
                tbl_bboxes = table.row
                img = numpy.reshape(img, newshape=pyr_shapes[l])

                for j in xrange(bboxes.shape[0]):
                    if l > 0:
                        resized_bbox = get_scaled_bbox(bboxes[j], pyr_shapes[l-1], pyr_shapes[l])
                        tbl_bboxes["picasaBatchNumber"] = resized_bbox[0]
                        tbl_bboxes["idxInPicasaBatch"] = resized_bbox[1]
                        tbl_bboxes["faceno"] = bboxes["faceno"][j]
                        tbl_bboxes["imgno"] = bboxes["imgno"][j]
                        tbl_bboxes["row"] = resized_bbox[2]
                        tbl_bboxes["col"] = resized_bbox[3]
                        tbl_bboxes["height"] = resized_bbox[4]
                        tbl_bboxes["width"] = resized_bbox[5]
                    else:
                        tbl_bboxes["picasaBatchNumber"] = bboxes["picasaBatchNumber"][j]
                        tbl_bboxes["idxInPicasaBatch"] = bboxes["idxInPicasaBatch"][j]
                        tbl_bboxes["faceno"] = bboxes["faceno"][j]
                        tbl_bboxes["imgno"] = bboxes["imgno"][j]
                        tbl_bboxes["row"] = bboxes["row"][j]
                        tbl_bboxes["col"] = bboxes["col"][j]
                        tbl_bboxes["height"] = bboxes["height"][j]
                        tbl_bboxes["width"] = bboxes["width"][j]

                    tbl_bboxes.append()

                if self.subsampling_rates is not None:
                    next_img = cv2.pyrDown(img, dstsize=pyr_shapes[l+1])
                else:
                    next_img = cv2.pyrDown(img)

                tmp_img = cv2.pyrUp(next_img, dstsize=pyr_shapes[l])
                l_img = numpy.reshape(img - tmp_img, newshape=numpy.prod(pyr_shapes[l]))

                table.flush()
                h5files[l].root.Data.X[i] = l_img
                img = next_img
                if l > 0:
                    bboxes = get_image_bboxes(i, table)
                h5files[l].flush()

        for h5file in h5files:
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

