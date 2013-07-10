import numpy
import os
import glob

def do_it(path, size, n_chan):
    files = glob.glob(os.path.join(path,'*.npy'))

    for f in files:
        print 'reading file', f
        orig = numpy.memmap(os.path.join(path, f), dtype=numpy.float32, mode='r')

        num_ok = 0
        
        lgt = len(orig)/(size[0]*size[1]*n_chan)
        for imgidx in xrange(lgt):
            img = orig[imgidx*size[0]*size[1]*n_chan:(imgidx+1)*size[0]*size[1]*n_chan]

            if img.max() != 0:
                num_ok += 1
        del orig

    if num_ok == 0:
        raise ValueError('{} contains only blank images.'.format(os.path.join(path, f)))

    for f in files:
        print 'processing file', f
        fname = f[:-4] + '_2.npy'
        out = numpy.memmap(os.path.join(path, fname), dtype=numpy.float32, mode='write', shape=(num_ok, size[0]*size[1]*n_chan))
        orig = numpy.memmap(os.path.join(path, f), dtype=numpy.float32, mode='r')
        lgt = len(orig)/(size[0]*size[1]*n_chan)

        idx = 0
        for imgidx in xrange(lgt):
            img = orig[imgidx*size[0]*size[1]*n_chan:(imgidx+1)*size[0]*size[1]*n_chan]

            if img.max() != 0:
                out[idx] = img
            idx += 1

        del out
        del orig
