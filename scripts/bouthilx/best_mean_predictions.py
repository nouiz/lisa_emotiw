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

#l = [('bagofmouth', 'pascal', 'samira', 'samira-audio'),
#('activity', 'bagofmouth', 'samira', 'samira-audio'),
#('activity', 'audio', 'bagofmouth', 'samira', 'samira-audio'),
#('activity', 'audio', 'bagofmouth', 'pascal', 'samira', 'samira-audio'),
#('activity', 'bagofmouth', 'pascal', 'samira', 'samira-audio'),
#('audio', 'bagofmouth', 'pascal', 'samira', 'samira-audio'),
#('activity', 'audio', 'bagofmouth', 'samira-audio'),
#('activity', 'audio', 'bagofmouth', 'pascal', 'samira-audio'),
#('activity', 'bagofmouth', 'pascal', 'samira-audio'),
#('bagofmouth', 'samira', 'samira-audio'),
#('audio', 'bagofmouth', 'samira', 'samira-audio'),
#('activity', 'bagofmouth', 'samira-audio'),
#('audio', 'bagofmouth', 'samira-audio'),
#('activity', 'pascal', 'samira', 'samira-audio'),
#('activity', 'audio', 'pascal', 'samira', 'samira-audio'),
#('activity', 'samira', 'samira-audio'),
#('activity', 'audio', 'bagofmouth', 'samira')]

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
    for mm in l:
        print mm
        mean_pred = mean([os.path.join(base_path,modes[mode],base_name % s) for mode in mm])
        if s!='test':
            rvals[mm] = np.mean(mean_pred.argmax(1)==targets)
        np.save("mean/"+"_".join(mm)+"_"+s,mean_pred)

    rvals2 = {}
    timer = Timer(len(mm)*3*7)
    timer.start()
    for j, [mm, rval] in enumerate(reversed(sorted(rvals.items(),key=lambda a:a[1]))):
        best = (None,0)
        paths = [os.path.join(base_path,modes[mode],base_name % s) for mode in mm]

        for k in range(3):
#                tests = np.random.multinomial(2000,[1/(0.+len(mm))]*len(mm),size=2000)/2000.
            tests = np.random.random((4000,len(mm),7))
#            tests = np.round(tests/np.sum(tests,axis=1)[:,None,:],decimals=2)
            tests = tests/np.sum(tests,axis=1)[:,None,:]
#            base = np.copy(best[i][0])#np.ones((len(mm),7))*1/(0.+len(mm))
            for test in tests:
                #print test
                #print base[:,i]
#                base[:,i] = test
                #print base
                mean_pred = take_best_p(paths,test)#base)
                if s!='test':
                    rval = np.mean(mean_pred.argmax(1)==targets)
                    if rval > best[1]:
                        best = (test,rval)#(base,rval)
            timer.print_update(1)

        
            print "round",k
            #print make_precision(best)
            print best[0]
            #rvals2[mm] = np.mean(take_best_p(paths,make_precision(best)).argmax(1)==targets)
            rvals2[mm] = best
            print rvals2[mm]
            print j,len(rvals)
    print "it took",timer.over()

    for mm, rval2 in reversed(sorted(rvals2.items(),key=lambda a:a[1][1])):
        print rval2[1],mm
        np.save('random_weights/'+s+'_'.join(mm),rval2[0])



#        mean_pred = take_best([os.path.join(base_path,modes[mode],base_name % s) for mode in mm])























