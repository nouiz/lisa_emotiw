from collections import OrderedDict
from extract_predictions import base_path, base_name, modes
from mean_predictions import mean, take_best
import numpy as np
import os
import sys

from emotiw.bouthilx.utils.timer import Timer

def take_best_p(preds_paths,precisions):
    shape = np.load(preds_paths[0]).shape
    preds = np.zeros(shape)
    efficiency = []

    for i, path in enumerate(preds_paths):
        preds += precisions[i]*np.load(path)#/precisions.sum(0)*np.load(path)

    return preds

weights = np.load(sys.argv[1])

mm = sys.argv[2:]

s = "train"

train_targets = np.load(os.path.join(base_path,"afew2_%s_targets.npy" % 'train'))
valid_targets = np.load(os.path.join(base_path,"afew2_%s_targets.npy" % 'valid'))
targets = np.concatenate((train_targets,valid_targets),axis=0)
#targets = valid_targets

train_paths = [os.path.join(base_path,modes[mode],base_name % 'train') for mode in mm]
valid_paths = [os.path.join(base_path,modes[mode],base_name % 'valid') for mode in mm]

train_mean_pred = take_best_p(train_paths,weights)
valid_mean_pred = take_best_p(valid_paths,weights)
mean_pred = np.concatenate((train_mean_pred,valid_mean_pred),axis=0)
#mean_pred = valid_mean_pred
rval = np.mean(mean_pred.argmax(1)==targets)
best = (weights,rval)
print "init"
print best[0]
print best[1]

r = 1.0

for k in range(10000):
    tests = np.random.normal(scale=0.10,size=(2000,len(mm),7))
    print "raw"
    print np.sum(tests > r)
    print np.sum(tests < -r)
    tests = (tests * (tests < r)) * (tests > -r)
    print "cliped"
    print np.sum(tests > r)
    print np.sum(tests < -r)
    for test in tests:
        new = test+weights 
        new = new * (new > 0)
#        print np.sum(new,axis=1)[:,None].shape
        new = np.round(new/np.sum(new,axis=0)[None,:],decimals=2)
        train_mean_pred = take_best_p(train_paths,new)
        valid_mean_pred = take_best_p(valid_paths,new)
        mean_pred = np.concatenate((train_mean_pred,valid_mean_pred),axis=0)
#        mean_pred = valid_mean_pred
        if s!='test':
            rval = np.mean(mean_pred.argmax(1)==targets)
            if rval > best[1]:
                best = (new,rval)

    print "round",k
    #print make_precision(best)
    print best[0]
    print best[1]
    #rvals2[mm] = np.mean(take_best_p(paths,make_precision(best)).argmax(1)==targets)

    np.save('random_weights/all_best_brute.npy',best[0])
    mean_pred = take_best_p(train_paths,best[0])
    np.save('random_weights/all_brute_train_predicts.npy',mean_pred)
    mean_pred = take_best_p(valid_paths,best[0])
    np.save('random_weights/all_brute_valid_predicts.npy',mean_pred)
    mean_pred = take_best_p([os.path.join(base_path,modes[mode],base_name % 'test') for mode in mm],best[0])
    np.save('random_weights/all_brute_test_predicts.npy',mean_pred)
