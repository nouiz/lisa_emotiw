import numpy
import os
import glob

def do_it(path, size, n_chan):
    files = glob.glob(os.path.join(path,'*.npy'))

    for f in files:
        print 'reading file', f
        orig = numpy.memmap(os.path.join(path, f), dtype=numpy.float32, mode='r')

        ok_idx = []
        
        lgt = len(orig)/(size[0]*size[1]*n_chan)
        for imgidx in xrange(lgt):
            img = orig[imgidx*size[0]*size[1]*n_chan:(imgidx+1)*size[0]*size[1]*n_chan]

            if img.max() != 0 and img.max() != float('nan'):
                ok_idx.append(imgidx)

        if len(ok_idx) == 0:
            #raise ValueError('{} contains only blank images.'.format(os.path.join(path, f)))
            print 'WARN: {} is empty'.format(f)
            del orig
            continue
        elif len(ok_idx) == lgt:
            print 'WARN: couldn\'t find blank images in {}'.format(f)

        print 'processing file', f
        fname = f[:-4] + '_2.npy'
        out = numpy.memmap(os.path.join(path, fname), dtype=numpy.float32, mode='write', shape=(len(ok_idx), size[0]*size[1]*n_chan))

        for idx, imgidx in enumerate(ok_idx):
            img = orig[imgidx*size[0]*size[1]*n_chan:(imgidx+1)*size[0]*size[1]*n_chan]

            out[idx] = img

        del out
        del orig
