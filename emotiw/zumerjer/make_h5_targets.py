import tables
import numpy as np
import sys


def make_targets(fname):
    data = tables.openFile(fname, mode='a')
    raw_targets = data.root.Data.y
    targets = data.createCArray(data.root.Data, '_y', atom = tables.Float32Atom(), shape = ((raw_targets.shape[0], raw_targets.shape[1], 96)))

    batch_size = 1000
    pixels = np.arange(0, 96)

    for idx in xrange(len(raw_targets)/batch_size + 1):

        y = np.cast['float32'](raw_targets[idx*batch_size:(idx+1)*batch_size,:])
        Y = np.zeros((y.shape[0], y.shape[1], 96))
        for i in xrange(y.shape[1]):
            Y[:,i,:] = np.where(y[:,i].reshape(y.shape[0],1)!= -1,
                    (np.exp(-(y[:,i].reshape(y.shape[0],1)-pixels)**2/(2*0.8**2)))/(np.sqrt(2*3.14159265359)*0.8),
                    -1)
        targets[idx*batch_size:(idx+1)*batch_size, :] = Y
        data.flush()

    data.removeNode(data.root.Data, "y", 1)
    data.renameNode(data.root.Data, "y", "_y")
    data.flush()


if __name__ == '__main__':
    make_targets(sys.argv[1])
