import numpy as np
import argparse
import os
import sys

def pca_whiten(data, var_fraction):
    """ principal components, retaining as many components as required to
        retain var_fraction of the variance

        Returns projected data, projection mapping, inverse mapping"""

    from numpy.linalg import eigh
    u, v = eigh(np.cov(data, rowvar=0))
    v = v[:, np.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<=u.sum()*var_fraction]
    u += 0.0000001
    numprincomps = u.shape[0]
    V = ( (u**(-0.5))[:numprincomps][None,:]*v[:,:numprincomps] ).T
    W = (u**0.5)[:numprincomps][None,:]*v[:,:numprincomps]
    return np.dot(data,V.T), V, W

def pca(data,dims):
    p = 1.0
    d = data.shape[1]
    while d!=dims:
        print p,d,dims
        data_white, pcamatrix, invpcamatrix = pca_whiten(data,p)
        d = data_white.shape[1]
        print dims,d
        p = p - 0.1*((d-dims)/(0.+dims+d))

    return data_white, pcamatrix, invpcamatrix

def process(path,dims):
    x = np.load(path)

    x = np.array(x).astype("float32")
    x -= x.mean(1)[:,None] #DC CENTERING
    x /= x.std(1)[:,None] + 0.001 #CONTRAST NORMALIZATION
    x -= x.mean(0)[None,:]
    x /= x.std(0)[None,:] + 0.001

    data_white, pcamatrix, invpcamatrix = pca(x,dims)
    return data_white

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims",type=int,required=True,help="Number of dimensions retained")
    parser.add_argument("--force",action='store_true',help="Force to overwrite npy file produced")
    parser.add_argument("data_paths",nargs="+",help="Npy files")
    options = parser.parse_args()

    dims = options.dims
    data_paths = options.data_paths

    for data_path in data_paths:
        data = process(data_path,dims)

        print data.shape
        print data[:5]

        pca_path = os.path.splitext(data_path)[0]+"_pca.npy"
        if os.path.isfile(pca_path) and not options.force:
            print """file \"%(file)s\" already exists. 
Use --force option if you wish to overwrite them""" % {"file":pca_path}
            parser.print_help()
            sys.exit(0)
        
        np.save(pca_path,data)


if __name__=="__main__":
    main()
