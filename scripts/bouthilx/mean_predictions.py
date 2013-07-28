import numpy as np
import argparse
import os

import emotiw.bouthilx.utils as utils
from extract_predictions import base_path, base_name

target_base_name = os.path.join(base_path,'afew2_%s_targets.npy')

target_names = {
    380: 'train',
    396: 'valid',
    312: 'test',
}

shortcut = {
    "activity": "ActivityRecognition",
    "samira":"ConvNetPlusSVM_PierreSamiraChris",
    "bag":"BagOfMouthFeatures",
    "mehdi":"MehdisModels",
    "pascal":"PascalAndXavier"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",default='mean',help="The directory where to extract predictions")
    parser.add_argument("--type",default='mean')
    parser.add_argument("predictions",nargs='+')

    options = parser.parse_args()

    out = options.out
    mean_type = options.type
    modes = options.predictions

    for s in ['train','valid','test']:
        s_preds_paths = [os.path.join(base_path,shortcut.get(mode,mode),base_name % s) for mode in modes]

        print s
        if mean_type=="mean":
            mean_preds = mean(s_preds_paths)
        elif mean_type=="take_best": 
            mean_preds = take_best(s_preds_paths)

        if s!='test':
            print 'score',np.mean(mean_preds.argmax(1)==np.load(target_base_name % s))*100,'%'

        if options.out is not None:
            np.save(os.path.splitext(options.out)[0]+'_%s.npy' % s,mean_preds)

def one_hot(targets):
    oh = np.zeros((targets.shape[0],7))
    for i, target in enumerate(targets):
        oh[i,target] = 1 
    return oh

def class_precision(preds_paths):
    shape = np.load(preds_paths[0]).shape
#    targets = np.load(target_base_name % 'valid')
    targets = np.load(target_base_name % 'train')
    targets = one_hot(targets)
    precisions = np.zeros((len(preds_paths),7))

    for i, path in enumerate(preds_paths):
        preds = np.load(path.replace('train_s','valid_s').replace('test_s','valid_s'))
        preds = np.load(path.replace('valid_s','train_s').replace('test_s','train_s'))
        preds_oh = one_hot(preds.argmax(1))
#        print np.sum((preds_oh==targets)*preds_oh,axis=0)
#        print np.sum(targets,axis=0)
#        print np.sum((preds_oh==targets)*preds_oh,axis=0)/np.sum(targets,axis=0)
#        print np.sum(targets*preds_oh)/np.sum(targets)
        precisions[i] = np.sum((preds_oh==targets)*preds_oh,axis=0)/np.sum(targets,axis=0)

    return precisions/precisions.sum(0)

def take_best(preds_paths):
    shape = np.load(preds_paths[0]).shape
    preds = np.zeros(shape)
    efficiency = []
    precisions = class_precision(preds_paths)
#    print precisions
#    print precisions.argmax(0)

    for i, path in enumerate(preds_paths):
#        print path
        print i, precisions[i]#/precisions.sum(0)
        preds += precisions[i]*np.load(path)#/precisions.sum(0)*np.load(path)

    return preds
    
def mean(preds_paths):
    shape = np.load(preds_paths[0]).shape
    
    preds = np.zeros([len(preds_paths)]+list(shape))

    for i, path in enumerate(preds_paths):
        preds[i] = np.load(path)

    mean = preds.mean(0)

    return mean

if __name__=="__main__":
    main()
