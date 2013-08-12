from collections import OrderedDict
from extract_predictions import base_path, base_name, modes
from mean_predictions import mean, take_best
import numpy as np
import os

from emotiw.bouthilx.utils.timer import Timer

mean = take_best
#del modes['samira-audio']

def take_best_p(preds_paths,precisions):
    shape = np.load(preds_paths[0]).shape
    preds = np.zeros(shape)
    efficiency = []
#    print precisions
#    print precisions.argmax(0)

    for i, path in enumerate(preds_paths):
#        print path
        preds += precisions[i]*np.load(path)#/precisions.sum(0)*np.load(path)

    return preds

tests = []

del modes['mehdi']
   
def assign(modes):
    l = []

    if len(modes)==0:
        return l

    
    for mode in modes:
        l.append((mode,))
        print l
        tmp = modes[:modes.index(mode)]+modes[modes.index(mode)+1:]
        print tmp
        assigned = assign(tmp)
        for mm in assigned:
            l.append(build([mode]+list(mm)))

    return unique(l)

def build(s):
    print s
    return tuple(sorted(s))

def unique(s):
    return OrderedDict.fromkeys(s).keys()

l = assign(sorted(modes.keys()))
print l

def make_precision(best):
    precision = np.zeros(best.values()[0][0].shape)

    for i in range(7):
        precision[:,i] = best[i][0][:,i]

    return precision
 
s = "valid"
rvals = {}
for s in ['train','valid','test']:

    for mm in l:

        paths = [os.path.join(base_path,modes[mode],base_name % s) for mode in mm]
        print mm
        weights = np.load('random_weights/'+'_'.join(mm)+'.npy')
        weights = np.round(weights,decimals=2)
        print weights
        mean_pred = take_best_p(paths,weights)
        np.save("mean/weights_from_val_"+"_".join(mm)+"_"+s,mean_pred)
