"""
Cython implementation of a Haar-like feature-based matching
for recovering the bounding box of a cropped face.  Roughly
a 270x speed up over NumPy.
"""
__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


__all__ = ["haar_like", "match_subregion"]


import numpy
cimport numpy

ctypedef numpy.float64_t DTYPE_t
ctypedef numpy.npy_intp INTP_t
from cython cimport view, boundscheck, wraparound


@wraparound(False)
@boundscheck(False)
cpdef inline DTYPE_t[:, :, :] _integral_image(DTYPE_t[:, :, :] x):
    """
    A faster, typed, inline-able version of _integral_image.
    """
    cdef INTP_t i, j, k
    cdef view.array out_arr = view.array(shape=(x.shape[0], x.shape[1],
                                                x.shape[2]),
                                         itemsize=sizeof(DTYPE_t),
                                         format="d", allocate_buffer=True)
    cdef DTYPE_t[:, :, :] out = out_arr
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                out[i, j, k] = x[i, j, k]
                if i > 0:
                    out[i, j, k] += out[i - 1, j, k]
                if j > 0:
                    out[i, j, k] += out[i, j - 1, k]
                if i > 0 and j > 0:
                    out[i, j, k] -= out[i - 1, j - 1, k]
    return out


@wraparound(False)
@boundscheck(False)
cdef inline void _integrate(DTYPE_t[:, :, :] image, INTP_t start_row,
                            INTP_t start_col, INTP_t end_row,
                            INTP_t end_col, DTYPE_t[:] ret):
    """
    A faster, typed, inline-able implementation of integration
    using an integral image.
    """
    cdef INTP_t k
    for k in range(image.shape[2]):
        ret[k] = image[end_row, end_col, k]
        if start_row > 0 and start_col > 0:
            ret[k] += image[start_row - 1, start_col - 1, k]
        if start_row > 0:
            ret[k] -= image[start_row - 1, end_col, k]
        if start_col > 0:
            ret[k] -= image[end_row, start_col - 1, k]


@wraparound(False)
@boundscheck(False)
cdef inline INTP_t _feature_storage(INTP_t order):
    """
    Calculate the number of features computed for a given value
    of `order`. `3\sum_{i=0}^K 4^i`.
    """
    cdef INTP_t s = 0, pow4 = 1
    for i in range(order + 1):
        s += pow4
        pow4 *= 4
    return 3 * s


def haar_like(numpy.ndarray int_im, start_row=0, start_col=0,
              end_row=None, end_col=None, order=0):
    """
    Compute Haar-like features from an integral image.

    Parameters
    ----------
    int_im : ndarray, 2- or 3-dimensional
        The integral image.

    start_row : int, optional
        The first row (inclusive) of the subregion on which to
        compute features. Defaults to the first row.

    start_col : int, optional
        The first column (inclusive) of the subregion on which to
        compute features. Defaults to the first column.

    end_row : int, optional
        The last row (exclusive) of the subregion on which to
        compute features. Defaults to the last row.

    end_col : int, optional
        The last column (exclusive) of the subregion on which to
        compute features. Defaults to the last column.

    order : int, optional
        Recurses this many times on a 2x2 grid and concatenate
        all the features.

    Returns
    -------
    features : ndarray, 1-dimensional
        A 1-dimensional array of features for this image. The
        length will depend on `order`.
    """
    if int_im.ndim == 2:
        int_im = int_im[:, :, numpy.newaxis]
    # TODO ERROR CHECK
    cdef view.array ret_arr = view.array((_feature_storage(order),
                                          int_im.shape[2]),
                                         itemsize=sizeof(DTYPE_t),
                                         format="d")
    cdef DTYPE_t[:, :] ret = ret_arr
    if int_im.ndim == 2:
        int_im = int_im[:, :, numpy.newaxis]
    end_row = int_im.shape[0] if end_row is None else end_row
    end_col = int_im.shape[0] if end_col is None else end_col

    cdef view.array tmp_arr = view.array(shape=(4 * int_im.shape[2],),
                                     itemsize=sizeof(DTYPE_t),
                                     format="d")
    cdef DTYPE_t[:] tmp = tmp_arr
    cdef DTYPE_t[:] lt = tmp[:int_im.shape[2]]
    cdef DTYPE_t[:] lb = tmp[int_im.shape[2]:2 * int_im.shape[2]]
    cdef DTYPE_t[:] rt = tmp[2 * int_im.shape[2]:3 * int_im.shape[2]]
    cdef DTYPE_t[:] rb = tmp[3 * int_im.shape[2]:]
    _haar_like(int_im, start_row, start_col, end_row, end_col, order, lt, lb,
               rt, rb, ret)
    return numpy.asarray(ret)


@wraparound(False)
@boundscheck(False)
cdef inline void _haar_like(DTYPE_t[:, :, :] int_im, INTP_t start_row,
                            INTP_t start_col, INTP_t end_row,
                            INTP_t end_col, int order, DTYPE_t[:] lt,
                            DTYPE_t[:] lb, DTYPE_t[:] rt, DTYPE_t[:] rb,
                            DTYPE_t[:, :] ret):
    """
    Low-level interface to the feature computation.
    """
    cdef INTP_t h_r = start_row + (end_row - start_row) // 2
    cdef INTP_t h_c = start_col + (end_col - start_col) // 2
    cdef INTP_t k, stride

    _integrate(int_im, start_row, start_col, h_r - 1, h_c - 1, lt)
    _integrate(int_im, h_r, start_col, end_row - 1, h_c - 1, lb)
    _integrate(int_im, start_row, h_c, h_r - 1, end_col - 1, rt)
    _integrate(int_im, h_r, h_c, end_row - 1, end_col - 1, rb)

    for k in range(int_im.shape[2]):
        ret[0, k] = lt[k] + rt[k] - lb[k] - rb[k]
        ret[1, k] = lt[k] - rt[k] + lb[k] - rb[k]
        ret[2, k] = lt[k] - rt[k] - lb[k] + rb[k]

    if order > 0:
        stride = _feature_storage(order - 1)
        _haar_like(int_im, start_row, start_col,
                   h_r - 1, h_c - 1, order - 1, lt, lb, rt, rb, ret[3:])
        _haar_like(int_im, start_row, h_c,
                   h_r - 1, end_col, order - 1, lt, lb, rt, rb, ret[3 + stride:])
        _haar_like(int_im, h_r, start_col,
                   end_row, h_c - 1, order - 1, lt, lb, rt, rb, ret[3 + 2 * stride:])
        _haar_like(int_im, h_r, h_c,
                   end_row, end_col, order - 1, lt, lb, rt, rb, ret[3 + 3 * stride:])


@boundscheck(False)
@wraparound(False)
def match_subregion(DTYPE_t[:, :, :] haystack, DTYPE_t[:, :, :] needle, int order=0):
    """
    Match a cropped portion of an image to determine
    its location in the original image, using Haar-like
    rectangular partial sums.

    Parameters
    ----------
    haystack : memoryview, 2-dimensional
        The image to search.

    needle : memoryview, 2-dimensional
        A cropped portion to be matched against
        subregions of `haystack`.

    order : int, optional
        The number of times to recursively apply the
        feature computation on smaller subregions
        (divided into a 2x2 grid at each stage). Default
        is 0.

    Returns
    -------
    best_i : int
        The topmost row of the best matching subregion.

    best_j : int
        The leftmost column of the best matching subregion.
    """
    if haystack.shape[2] != needle.shape[2]:
        raise ValueError("needle and haystack must have same third dimension")
    elif (haystack.shape[0] < needle.shape[0] or
          haystack.shape[1] < needle.shape[1]):
        raise ValueError("spatial dimensions of needle exceed haystack's")

    cdef DTYPE_t[:, :, :] haystack_int = _integral_image(haystack)
    cdef DTYPE_t[:, :, :] needle_int = _integral_image(needle)

    # Buffer and memoryview for intermediate computation of
    # features.
    cdef INTP_t storage = _feature_storage(order)
    cdef view.array retval = view.array(shape=(storage, haystack.shape[2]),
                                        itemsize=sizeof(DTYPE_t),
                                        allocate_buffer=True,
                                        format="d")
    cdef DTYPE_t[:, :] these = retval

    # Features for the sought subregion.
    cdef view.array needle_feat_arr = view.array(shape=(storage,
                                                        needle.shape[2]),
                                                 itemsize=sizeof(DTYPE_t),
                                                 allocate_buffer=True,
                                                 format="d")
    cdef DTYPE_t[:, :] needle_feat = needle_feat_arr
    cdef view.array tmp_arr = view.array(shape=(4 * haystack.shape[2],),
                                         itemsize=sizeof(DTYPE_t),
                                         format="d")
    cdef DTYPE_t[:] tmp = tmp_arr
    cdef DTYPE_t[:] lt = tmp[:haystack.shape[2]]
    cdef DTYPE_t[:] lb = tmp[haystack.shape[2]:2 * haystack.shape[2]]
    cdef DTYPE_t[:] rt = tmp[2 * haystack.shape[2]:3 * haystack.shape[2]]
    cdef DTYPE_t[:] rb = tmp[3 * haystack.shape[2]:]

    _haar_like(needle_int, 0, 0, needle_int.shape[0], needle_int.shape[1],
               order, lt, lb, rt, rb, needle_feat)
    cdef INTP_t i, j, best_i, best_j, r, c, m, k
    cdef DTYPE_t best, score
    r, c = needle.shape[0], needle.shape[1]
    best = numpy.inf
    for i in xrange(0, haystack.shape[0] + 1 - r):
        for j in xrange(0, haystack.shape[1] + 1 - c):
            _haar_like(haystack_int, i, j, i + r, j + c, order, lt, lb, rt, rb, these)
            score = 0
            for m in range(these.shape[0]):
                for k in range(these.shape[1]):
                    score += (these[m, k] - needle_feat[m, k]) ** 2
            if score < best:
                best_i = i
                best_j = j
                best = score
    return best_i, best_j
