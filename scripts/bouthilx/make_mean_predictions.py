import numpy as np
import argparse
import os

import emotiw.bouthilx.utils as utils

classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",help="The directory where to extract predictions")
    parser.add_argument("--data",action='append',required=True,help="Data path (.npy)")
    parser.add_argument("--targets",action='append',help="Target path (.npy)")
    parser.add_argument("--normalize",default=False)

    parser.add_argument("model_paths",nargs='+',help="Pylearn2 models")

    options = parser.parse_args()

    out = options.out
    data_paths = options.data
    model_paths = options.model_paths
    normalize = "True"==options.normalize
 
    print model_paths
    print data_paths

    datas = [(data_path, np.load(data_path)) for data_path in data_paths]

    from theano import config
    from theano import function
    from pylearn2.utils import serial

    from emotiw.bouthilx.datasets import FeaturesDataset
    print np.mean(datas[0][1] == FeaturesDataset(features_paths= ["audio/test26_best_audio_mlp_valid_feats_features.npy",
                                                          "pascal/afew2_valid_rmananinpic_emmanuel_features.npy"],
                               targets_path= "pascal/afew2_valid_rmananinpic_emmanuel_targets.npy",
                               base_path= "/data/afew",
                               normalize= True,
                               shuffle= False).get_design_matrix())

    predictions = []

    if len(datas)<len(model_paths) and len(datas)==1:
        l = []
        for model_path in model_paths:
            l.append((model_path,datas[0]))

        iterator = l
    else:
        iterator = zip(model_paths,datas)

    for model_path, (data_path, data) in iterator:
        
        model = serial.load(model_path)

        batch_size = model.get_test_batch_size()

        X = model.get_input_space().make_batch_theano()
        f = function([X],model.fprop(X))

        if normalize:
            # set 0s to mean
            tmp = data[:]
            tmp -= tmp.min(0)
            tmp /= (tmp==0) + tmp.max(0)
            # range is [-1,1]
            tmp = tmp*2.0-1.0 
            tmp = (data.min(0)!=data.max(0))*tmp # set empty dimensions to 0
            data = tmp

#        print data
        y_hat = utils.apply(f,data,batch_size)
#        print y_hat.sum(1)

        if options.targets is not None:
            targets = np.load(options.targets[0])
#            print targets
#            print y_hat.argmax(1)
#            j = 0
#            for i, target in enumerate(targets):
#                j += classes[int(target)]==classes[y_hat[i].argmax()]
#                print classes[int(target)], classes[y_hat[i].argmax()], j, len(targets) 
#
#            print np.mean(y_hat.argmax(1)==targets)
        predictions.append(y_hat)

    mean = np.array(predictions).mean(0)

    if options.targets is not None:
        targets = np.load(options.targets[0])
#        print mean.argmax(1)
#        print targets
        print np.mean(mean.argmax(1)==targets)
        j = 0
#        for i, target in enumerate(targets):
#            j += classes[int(target)]==classes[mean[i].argmax()]
#            print classes[int(target)], classes[mean[i].argmax()], j, len(targets) 

    if options.out is not None:
        np.save(options.out,mean)

if __name__=="__main__":
    main()
