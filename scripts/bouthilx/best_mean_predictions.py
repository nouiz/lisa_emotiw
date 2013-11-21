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

#del modes['mehdi']
   
def assign(modes):
    l = []

    if len(modes)==0:
        return l

    
    for mode in modes:
        l.append((mode,))
        tmp = modes[:modes.index(mode)]+modes[modes.index(mode)+1:]
        assigned = assign(tmp)
        for mm in assigned:
            l.append(build([mode]+list(mm)))

    return unique(l)

def build(s):
    return tuple(sorted(s))

def unique(s):
    return OrderedDict.fromkeys(s).keys()

#l = assign(sorted(modes.keys()))

l = [('bagofmouth', 'pascal', 'samira', 'samira-audio'),
('activity', 'bagofmouth', 'samira', 'samira-audio'),
('activity', 'audio', 'bagofmouth', 'samira', 'samira-audio'),
('activity', 'audio', 'bagofmouth', 'pascal', 'samira', 'samira-audio'),
('activity', 'bagofmouth', 'pascal', 'samira', 'samira-audio'),
('audio', 'bagofmouth', 'pascal', 'samira', 'samira-audio'),
('activity', 'audio', 'bagofmouth', 'samira-audio'),
('activity', 'audio', 'bagofmouth', 'pascal', 'samira-audio'),
('activity', 'bagofmouth', 'pascal', 'samira-audio'),
('bagofmouth', 'samira', 'samira-audio'),
('audio', 'bagofmouth', 'samira', 'samira-audio'),
('activity', 'bagofmouth', 'samira-audio'),
('audio', 'bagofmouth', 'samira-audio'),
('activity', 'pascal', 'samira', 'samira-audio'),
('activity', 'audio', 'pascal', 'samira', 'samira-audio'),
('activity', 'samira', 'samira-audio'),
('audio', 'samira'),
('activity', 'audio', 'bagofmouth', 'samira')]

l = [('audio','samira'),
('activity', 'audio', 'bagofmouth', 'samira')]

def make_precision(best):
    precision = np.zeros(best.values()[0][0].shape)

    for i in range(7):
        precision[:,i] = best[i][0][:,i]

    return precision
 
s = "valid"
rvals = {}
for s in ['valid']:#'train','valid','test']:

    if s!='test':
        targets = np.load(os.path.join(base_path,"afew2_%s_targets.npy" % s))

    for j, mm in enumerate(l):
        rvals[mm] = (None,0)
        paths = [os.path.join(base_path,modes[mode],base_name % s) for mode in mm]

        for k in range(5):
            tests = np.random.random((4000,len(mm),7))
            tests = tests/np.sum(tests,axis=1)[:,None,:]
            for test in tests:
                mean_pred = take_best_p(paths,test)
                if s!='test':
                    rval = np.mean(mean_pred.argmax(1)==targets)
                    if rval > rvals[mm][1]:
                        rvals[mm] = (test,rval)#(base,rval)
        
            print "round",k
            #print rvals[mm][0]
            print mm
            print rvals[mm][1]
            print j,len(l)

    for mm, rval in reversed(sorted(rvals.items(),key=lambda a:a[1][1])):
        print rval[1],mm
        np.save('random_weights/'+s+'_'.join(mm),rval[0])

    y = 0
    while True:
        for mm in l:
            train_paths = [os.path.join(base_path,modes[mode],base_name % 'train') for mode in mm]
            valid_paths = [os.path.join(base_path,modes[mode],base_name % 'valid') for mode in mm]

            targets = np.load(os.path.join(base_path,"afew2_%s_targets.npy" % 'valid'))

            paths = [os.path.join(base_path,modes[mode],base_name % 'valid') for mode in mm]

            mean_pred = take_best_p(paths,rvals[mm][0])
            rval = np.mean(mean_pred.argmax(1)==targets)
            #print "init"
            #print rvals[mm][0]
            #print rvals[mm][1]

            r = 1.0

            for k in range(3):
                tests = np.random.normal(scale=0.10,size=(2000,len(mm),7))
                tests = (tests * (tests < r)) * (tests > -r)
                for test in tests:
                    new = test+rvals[mm][0]
                    new = new * (new > 0)
                    new = np.round(new/np.sum(new,axis=0)[None,:],decimals=2)
                    mean_pred = take_best_p(paths,new)
                    if s!='test':
                        rval = np.mean(mean_pred.argmax(1)==targets)
                        if rval > rvals[mm][1]:
                            rvals[mm] = (new,rval)

                np.save('random_weights/best_both'+'_'.join(mm)+'.npy',rvals[mm][0])
                mean_pred = take_best_p(train_paths,rvals[mm][0])
                np.save('random_weights/brute_train_predicts'+'_'.join(mm)+'.npy',mean_pred)
                mean_pred = take_best_p(valid_paths,rvals[mm][0])
                np.save('random_weights/brute_valid_predicts'+'_'.join(mm)+'.npy',mean_pred)
                mean_pred = take_best_p([os.path.join(base_path,modes[mode],base_name % 'test') for mode in mm],rvals[mm][0])
                np.save('random_weights/brute_test_predicts'+'_'.join(mm)+'.npy',mean_pred)

            print mm

        y += 1
        print "round", y

        for mm, rval in reversed(sorted(rvals.items(),key=lambda a:a[1][1])):
            print rval[1],mm

















