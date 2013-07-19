import argparse
import numpy as np
<<<<<<< HEAD
import theano
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets.preprocessing import Standardize
import emotiw.common.datasets.faces.afew2 as afew2
import emotiw.common.aggregation.pooling as pooling
=======
>>>>>>> 37bef88e556862d180581c89fe7b7b5017036991

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",nargs=1)
    parser.add_argument("model_path", nargs=1)
    options = parser.parse_args()

    out = options.out[0]
<<<<<<< HEAD
    out = out.strip('.npy')
    model_path = options.model_path[0]

    try:
        model = serial.load(model_path)
    except Exception, e:
        print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
        print e

    dataset = yaml_parse.load(model.dataset_yaml_src.replace('/data/afew/facetubes/p10','/data/lisatmp/bouthilx/facetubes/p10'))
    preprocessor = Standardize()
    preprocessor.apply(dataset.raw,can_fit=True)
    X = dataset.raw.get_design_matrix()
    X = X.reshape(X.shape[0],3,96,96).transpose(0,2,3,1)
#    X = X.reshape(X.shape[0],96,96,3)
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std_eps = 1e-4
    print mean.shape
    print std.shape
#    print preprocessor._mean, mean
#    print preprocessor._std,std

    class DummyDataset:
        def __init__(self,X):
            self.X = X.transpose(0,3,1,2).reshape(X.shape[0],np.prod(X.shape[1:]))

        def get_design_matrix(self):
            return self.X

        def set_design_matrix(self,X):
            self.X = X

    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)

    f = theano.function([X],Y)
    X = theano.tensor.tensor3()
    pool = theano.function([X],pooling.max_pool(X))

    Xs = []
    y = []

    for i, (x,y) in enumerate(zip(dataset.raw.get_design_matrix(),dataset.raw.get_targets())):
        print i, dataset.raw.get_design_matrix().shape
#        Xs.append(f(np.cast['float32'](row[None,:].reshape(1,3,96,96).transpose(0,2,3,1))))
        Xs.append(f(np.cast['float32'](x[None,:].reshape(1,96,96,3))))
        y.append(y)
        print Xs[-1]
        if i==5:
            break

    Xs = np.array(Xs)
    y = np.array(y)
    print "Xs"
    print Xs.shape
    print y.shape
    print Xs.mean(0)
    print Xs.sum(0)
    print (Xs.argmax(1)==y.argmax(1)).mean()

    d = afew2.AFEW2ImageSequenceDataset()
    fichier = open('log','w')
    for i, [clip, info, target] in enumerate(zip(d.imagesequences,d.seq_info,d.labels)):
        print i
        
        results = []
        for facetube in d.get_facetubes(i):
            onehot = np.zeros(len(d.emotionNames.keys()))
            onehot[d.emotionNames.values().index(target)] = 1.0
#            targets[info[0]] += [onehot]*facetube.shape[0]
            tmp = []
            for face_i in xrange(0,len(facetube),model.get_test_batch_size()):
#                print face_i, min(len(facetube),face_i+model.get_test_batch_size()), len(facetube)
                dd = DummyDataset(facetube[face_i:min(len(facetube),face_i+model.get_test_batch_size())])
                dd = facetube[face_i:min(len(facetube),face_i+model.get_test_batch_size())]
                dd = (dd-mean)/(std_eps+std)
#                preprocessor.apply(dd,can_fit=False)
#                tmp += list(f(np.cast['float32'](dd.X.reshape(dd.X.shape[0],3,96,96).transpose(0,2,3,1))))
                tmp += list(f(np.cast['float32'](dd)))
#                fichier.write(str(i)+' tmp\n')
#                fichier.write(str(tmp[-1])+"\n")
                print tmp[-1]
            results += tmp#results).mean(0))
#            fichier.write('results\n')
#            fichier.write(str(results[-1])+"\n")

#            print results[-1].shape
        if len(results):
            Xs.append(pool(np.array(results)[None,:,:].transpose(1,0,2))[0])
#            fichier.write('Xs\n')
#            fichier.write(str(Xs[-1])+"\n")

    #        print Xs[-1].shape
    #        print Xs[-1], onehot
            y.append(onehot)
#        if i>=5:
#            break
    fichier.close()
    Xs = np.array(Xs)
    y = np.array(y)
    print "Xs"
    print Xs.shape
    print y.shape
    print Xs.mean(0)
    print Xs.sum(0)
    print (Xs.argmax(1)==y.argmax(1)).mean()
    np.save(out+"_X.npy",Xs)
=======

    d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)
    for i, [clip, info, target] in enumerate(zip(d.imagesequences,d.seq_info,d.labels)):
        
        for facetube in d.get_facetubes(i):
            onehot = np.zeros(len(d.emotionNames.keys()))
            onehot[d.emotionNames.values().index(target)] = 1.0
            targets[info[0]] += [onehot]*facetube.shape[0]
            faces[info[0]] += [face for face in facetube]
 
    np.save(out+"_X.npy",X)
>>>>>>> 37bef88e556862d180581c89fe7b7b5017036991
    np.save(out+"_y.npy",y)

if __name__ == "__main__":
    main()
