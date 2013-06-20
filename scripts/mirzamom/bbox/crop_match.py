"""
NumPy-based implementation of a Haar-like feature-based matching
for recovering the bounding box of a cropped face.
"""
__author__ = "David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__email__ = "wardefar@iro"
__maintainer__ = "David Warde-Farley"


__all__ = ["haar_like", "match_subregion"]


import numpy

# _integral_image() and _integrate() are copyright (c) the scikit-image team.
# Released under the 3-clause BSD license.

def _integral_image(x):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    x : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image / summed area table.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    return x.cumsum(1).cumsum(0)


def _integrate(ii, r0, c0, r1, c1):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Integral (sum) over the given window.
    """
    S = 0

    S += ii[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += ii[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= ii[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= ii[r1, c0 - 1]

    return S


def haar_like(int_im, start_row=0, start_col=0, end_row=None,
              end_col=None, order=0):
    """
    Compute Haar-like features from an integral image.

    Parameters
    ----------
    int_im : ndarray, ndim >= 2
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
    if end_row is None:
        end_row = int_im.shape[0]
    if end_col is None:
        end_col = int_im.shape[1]

    h_r = start_row + (end_row - start_row) // 2
    h_c = start_col + (end_col - start_col) // 2
    lt = _integrate(int_im, start_row, start_col, h_r - 1, h_c - 1)
    lb = _integrate(int_im, h_r, start_col, end_row - 1, h_c - 1)
    rt = _integrate(int_im, start_row, h_c, h_r - 1, end_col - 1)
    rb = _integrate(int_im, h_r, h_c, end_row - 1, end_col - 1)
    f = [lt + rt - lb - rb,
         lt - rt + lb - rb,
         lt - rt - lb + rb]
    if order > 0:
        f.extend(haar_like(int_im, start_row, start_col,
                                 h_r - 1, h_c - 1, order - 1))
        f.extend(haar_like(int_im, start_row, h_c,
                                 h_r - 1, end_col, order - 1))
        f.extend(haar_like(int_im, h_r, start_col,
                                 end_row, h_c - 1, order - 1))
        f.extend(haar_like(int_im, h_r, h_c,
                                 end_row, end_col, order - 1))
    return numpy.asarray(f)


def match_subregion(haystack, needle, order=0):
    """
    Match a cropped portion of an image to determine
    its location in the original image, using Haar-like
    rectangular partial sums.

    Parameters
    ----------
    haystack : ndarray, ndim >= 2
        The image to search.

    needle : ndarray, ndim >= 2
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
    needle_int = _integral_image(needle)
    needle_feat = haar_like(needle_int, order=order)
    haystack_int = _integral_image(haystack)
    r, c = needle.shape[:2]
    best = numpy.inf
    best_i = numpy.inf
    best_j = numpy.inf
    for i in xrange(0, haystack.shape[0] + 1 - r):
        for j in xrange(0, haystack.shape[1] + 1 - c):
            these = haar_like(haystack_int, i, j, i + r, j + c,
                              order=order)
            score = ((these - needle_feat) ** 2).sum()
            if score < best:
                best_i = i
                best_j = j
                best = score
    return best_i, best_j


_haar_like_py = haar_like
_match_subregion_py = match_subregion


try:
    import _crop_match
    haar_like = _crop_match.haar_like
    match_subregion = _crop_match.match_subregion
except ImportError:
    print 'failed loading cython'
