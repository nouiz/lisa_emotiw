import numpy
import math
import tables
import os

numKeypoints = 98


def perturb(input_file, output_file, maxTrans = 3, maxRotDegree = 2, scaling=True, batch_size=1, repeat=4):
    Mat = numpy.eye(3,3)

    inp = tables.openFile(input_file, mode='r')
    data_in = inp.root.Data.X

    out = tables.openFile(output_file, mode='w')

    dat = out.createGroup('/', 'Data', 'Data')
    data_x = out.createCArray(dat, atom=tables.UInt8Atom(), name='X', shape=((repeat*len(data_in), data_in.shape[1], data_in.shape[2], data_in.shape[3])))
    data_y = out.createCArray(dat, atom=tables.UInt8Atom(), name='y', shape=((repeat*len(data_in), numKeypoints*2)))

    data = data_x
    labels = data_y
    labels_in = inp.root.Data.y

    assert len(data) % batch_size == 0

    mapping = numpy.array([(x, y, 1) for x in range(96) for y in range(96)])

    for repetition in range(repeat):
        for idx in xrange(len(data)/batch_size):
            print idx
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
                scaleFactor = 1 + numpy.random.random((1)) * 0.1
                scaleMat[0,0] = scaleMat[1,1] = scaleFactor
                perturbMat = numpy.dot(rotMat, numpy.dot(scaleMat,transMat))
            else:
                perturbMat = numpy.dot(rotMat, transMat)

            final_mapping = numpy.dot(mapping, perturbMat)[:,0:2]
            final_mapping = numpy.array(numpy.round(final_mapping))
            for an_idx, x in enumerate(final_mapping):
                if not 96 > x[0] >= 0 or not 96 > x[1] >= 0:
                    final_mapping[an_idx] = 96

            final_mapping = numpy.cast['uint8'](final_mapping)

            #if batch_size == 1:
            #    get = data[idx*batch_size, final_mapping[:,0], final_mapping[:,1], :]
            #    sett = data_in[idx*batch_size]
            #else:
            #    get = data[idx*batch_size:(idx+1)*batch_size, final_mapping[:,0], final_mapping[:,1], :]
            #    sett = data_in[idx*batch_size:(idx+1)*batch_size]

            for curr, f in enumerate(final_mapping):
                if numpy.all(f < 96):
                    if batch_size == 1:
                        data[idx, f[0], f[1], :] = data_in[idx, curr//96, curr%96, :]
                    else:
                        data[idx*batch_size:(idx+1)*batch_size, f[0], f[1], :] = \
                                data_in[idx*batch_size:(idx+1)*batch_size]

            for pt_idx in xrange(numKeypoints):
                if batch_size == 1:
                    l = numpy.cast['int8'](labels_in[idx].reshape(numKeypoints, 2)[pt_idx])
                    if numpy.all(l >= 0) and numpy.all(l < 96):
                        #print final_mapping.reshape(96,96,2)[l[0], l[1]]
                        #print l[0], l[1]
                        #print pt_idx*2, (pt_idx+1)*2
                        labels[idx, pt_idx*2:(pt_idx+1)*2] = final_mapping.reshape(96,96,2)[l[0], l[1]]
                else:
                    labels[idx*batch_size:(idx+1)*batch_size][pt_idx] = final_mapping[numpy.cast['uint8'](labels_in[idx*batch_size:(idx+1)*batch_size])][pt_idx//2 + pt_idx%2]
                    #XXX

        #data_x = out.createCArray(dat, atom=tables.UInt8Atom(), name='X_', shape=((repeat*len(data_in), data_in.shape[1], data_in.shape[2], data_in.shape[3])))
        #data_y = out.createCArray(dat, atom=tables.Float32Atom(), name='y_', shape=((repeat*len(data_in), numKeypoints*2, 96)))

        #data_x = data[:, :-1, :-1, :]
        #data_y = labels[:, :, :-1]

        #out.removeNode(dat, 'y')
        #out.renameNode(dat, 'y', 'y_')
        #out.removeNode(dat, 'X')
        #out.renameNode(dat, 'X', 'X_')

        data_x.flush()
        data_y.flush()

        out.close()
        inp.close()
