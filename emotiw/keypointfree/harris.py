"""
harris corners 

adapted from http://www.janeriksolem.net/2009/01/harris-corner-detector-in-python.html
"""



from scipy import *
from scipy import signal


def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size+1, -sizey:sizey+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 

    return gx,gy


def gauss_kernel(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def gauss_derivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian 
        derivative filters of size n. The optional argument 
        ny allows for a different size in the y direction."""

    gx,gy = gauss_derivative_kernels(n, sizey=ny)

    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')

    return imx,imy


def compute_harris_response(image):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""

    #derivatives
    imx,imy = gauss_derivatives(image, 3)

    #kernel for blurring
    gauss = gauss_kernel(3)

    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_distance=3, border=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating 
        corners and image boundary"""

    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]

    #sort candidates
    index = argsort(candidate_values)

    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[border:-border,border:-border] = 1

    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),(coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    import pylab

    pylab.figure()
    pylab.gray()
    pylab.imshow(image)
    pylab.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
    pylab.axis('off')
    pylab.show()


def get_corners(image, min_dist=10, threshold=0.1):
    return get_harris_points(compute_harris_response(image), min_dist, threshold)


