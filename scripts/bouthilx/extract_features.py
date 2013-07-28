import numpy as np
import argparse
import os
import emotiw.bouthilx.utils as utils
from theano import config
from theano import function
from pylearn2.utils import serial



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True,help="The directory where to extract features")
    parser.add_argument("--layer",default=None,type=int,help="Layer to extract (default is prediction of the model)")
    parser.add_argument("--data",action='append',required=True,help="Data path (.npy)")
    parser.add_argument("--targets",action='append',help="Target path (.npy)")
    parser.add_argument("--normalize",default=True)

    parser.add_argument("model_paths",nargs='+',help="Pylearn2 models")

    options = parser.parse_args()

    out = options.out
    layer_idx = options.layer
    data_paths = options.data
    model_paths = options.model_paths
    normalize = options.normalize
    
    print model_paths
    print data_paths

    if options.targets:
        datas = [(data_path, 
                  np.cast[config.floatX](np.load(data_path)), 
                  np.cast[config.floatX](np.load(target_path))) for data_path, target_path in zip(data_paths,options.targets)]
    else:
        datas = [(data_path, np.cast[config.floatX](np.load(data_path)), None) for data_path in data_paths]

#    if normalize:
#        # set 0s to mean
#        tmp = X[:]
#        tmp -= tmp.min(0)
#        tmp /= (tmp==0) + tmp.max(0)
#        # range is [-1,1]
#        tmp = tmp*2.0-1.0 
#        tmp = (X.min(0)!=X.max(0))*tmp # set empty dimensions to 0
#        X = tmp

    for model_path in model_paths:
        for data_path, data, targets in datas:
            features = []

            if normalize:
                # set 0s to mean
                tmp = data[:]
                tmp -= tmp.min(0)
                tmp /= (tmp==0) + tmp.max(0)
                # range is [-1,1]
                tmp = tmp*2.0-1.0 
                tmp = (data.min(0)!=data.max(0))*tmp # set empty dimensions to 0
                data = tmp

            features = get_features(model_path,data,layer_idx)

            if targets is not None:
                print data_path
#                print features.sum(1)
                print np.mean(features.argmax(1)==targets)

            tmpout = os.path.splitext(model_path.split('/')[-1])[0]
            tmpout += "_"+os.path.splitext(data_path.split('/')[-1])[0]
            if layer_idx is None:
                tmpout += "_predictions.npy"
            else:
                tmpout += "_features.npy"
            print os.path.join(out,tmpout)
            np.save(os.path.join(out,tmpout),features)

def get_features(model_path,data,layer_idx=None):

    model = serial.load(model_path)

    batch_size = model.get_test_batch_size()

    X = model.get_input_space().make_batch_theano()
    rval = X

    if layer_idx is None:
        layers = model.layers
    else:
        layers = model.layers[:layer_idx]

    for layer in layers:
        rval = layer.fprop(rval)

    f = function([X], rval)
    
    return utils.apply(f,data,batch_size)

if __name__=="__main__":
    main()
