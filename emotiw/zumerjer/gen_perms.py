import numpy
import math
import tables
import os
import Image
import sys

numKeypoints = 98


def perturb(input_file, output_file, maxTrans = 15, maxRotDegree = 7, scaling=True, batch_size=1, repeat=4):
    Mat = numpy.eye(3,3)

    inp = tables.openFile(input_file, mode='r')
    data_in = inp.root.Data.X

    out = tables.openFile(output_file, mode='w')

    dat = out.createGroup('/', 'Data', 'Data')
    data_x = out.createCArray(dat, atom=tables.UInt8Atom(), name='X', shape=(((repeat+1)*len(data_in), data_in.shape[1], data_in.shape[2], data_in.shape[3])))
    data_y = out.createCArray(dat, atom=tables.UInt8Atom(), name='y', shape=(((repeat+1)*len(data_in), numKeypoints*2)))

    data = data_x
    labels = data_y
    labels_in = inp.root.Data.y

    assert len(data) % batch_size == 0

    data[:len(data_in), :] = data_in[:]
    labels[:len(data_in), :] = labels_in[:]

    data.flush()
    labels.flush()

    the_idx = 0
    next_idx = len(labels_in)

    for repetition in range(repeat):
        print 'Pass #' + str(repetition + 1), '...'
        for idx in xrange(len(labels_in)/batch_size):
            the_idx = next_idx
            next_idx = the_idx + batch_size

          #tranlation
            transMat = Mat.copy()
            transMat[0,2] = maxTrans * (numpy.random.random((1))-0.5)
            transMat[1,2] = maxTrans * (numpy.random.random((1))-0.5)

          #rotation
            rotMat = Mat.copy()
            theta = (numpy.random.random((1))-0.5) * 3.14159265359*(maxRotDegree/180.0)
            rotMat[0:2,0:2] = [[math.cos(theta), math.sin(theta)],[-1 * math.sin(theta), math.cos(theta)]]

          #scaling
            if scaling:
                scaleMat = Mat.copy()
                scaleFactor = 1 + numpy.random.random((1)) * 0.3
                scaleMat[0,0] = scaleMat[1,1] = scaleFactor
                perturbMat = numpy.dot(rotMat, numpy.dot(scaleMat,transMat))
            else:
                perturbMat = numpy.dot(rotMat, transMat)

            perturbSubMat = perturbMat[0:2,:]
            perturbSubMat = [x for y in perturbSubMat for x in y]

            for i in xrange(batch_size):
                data[the_idx + i] = numpy.array(
                        Image.fromarray(numpy.cast['uint8'](data_in[idx*batch_size + i]
                            )).transform((96,96), Image.AFFINE, perturbSubMat).getdata(),
                        dtype='float32').reshape((batch_size, 96, 96, 3))

            if batch_size == 1:
                cut = the_idx
                cut_in = idx
            else:
                cut = slice(the_idx, next_idx, None)
                cut_in = slice(idx*batch_size, (idx+1)*batch_size, None)

            lbl = labels_in[cut_in].reshape(batch_size, numKeypoints, 2)
            labels[cut] = numpy.dot(lbl.reshape(numKeypoints, 2), perturbMat[0:2,:])[:,0:2].round().reshape(batch_size, numKeypoints*2)

        data_x.flush()
        data_y.flush()

        print 'Pass #' + str(repetition + 1), ': Done!'

    out.close()
    inp.close()

if __name__ == '__main__':
    perturb(sys.argv[1], sys.argv[2])
