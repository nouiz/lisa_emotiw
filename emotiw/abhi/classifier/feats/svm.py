from sklearn import svm
import numpy
from pylearn2.utils import serial
from sklearn import preprocessing
from emotiw.common.datasets.faces.faceimages import basic_7emotion_names
import math
import os

#train
train = serial.load('train.pkl')

train_x, train_y, train_path = train['x'], train['y'], train['path']

#train_x, train_y = [], []

#for key in train:
#    train_x.append(train[key][0])
#    train_y.append(train[key][1])

#print train_x, train_y


#validation
valid = serial.load('val.pkl')
valid_x, valid_y, valid_path = valid['x'], valid['y'], valid['path']
#valid_x, valid_y = [], []

#for key in valid:
#    valid_x.append(valid[key][0])
#    valid_y.append(valid[key][1])


#test
test = serial.load('test.pkl')
test_x, test_path = test['x'], test['path']



#train_X = preprocessing.normalize(preprocessing.normalize(preprocessing.scale(train_x), axis = 0), axis = 1)
#valid_x = preprocessing.normalize(preprocessing.normalize(preprocessing.scale(valid_x), axis = 0), axis = 1)

train_x = preprocessing.normalize(preprocessing.scale(train_x))
valid_x = preprocessing.normalize(preprocessing.scale(valid_x))

#valid_x = preprocessing.scale(preprocessing.normalize(valid_x))
#train_x = preprocessing.scale(preprocessing.normalize(train_x))

#train_X = preprocessing.binarize(preprocessing.scale(train_x))
#valid_x = preprocessing.binarize(preprocessing.scale(valid_x))

#scaleObj  = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#train_X = preprocessing.normalize(scaleObj.transform(train_x))
#valid_x = preprocessing.normalize(scaleObj.transform(valid_x))


#c_vals = numpy.logspace(math.log10(0.01), math.log10(0.03), num = 100)
c_vals = 0.023,
for c in c_vals:
    #tols = numpy.logspace(math.log10(5), math.log10(15), 10)
    tols = 6.383,
    for tol in tols:
        clf = svm.LinearSVC(C = c, tol=tol)
        #clf = svm.SVC(C = c, kernel='poly', degree=2)
        #clf = svm.NuSVC(nu=c)
        clf.fit(train_x, train_y)
        print c, tol, clf.score(valid_x, valid_y), clf.score(train_x, train_y)
        #print c, clf.predict(test_x)

# prediction
pred_train = clf.predict(train_x)
pred_valid = clf.predict(valid_x)
pred_test = clf.predict(test_x)

#scores
emotion_train = clf.decision_function(train_x)
emotion_test = clf.decision_function(test_x)
emotion_valid = clf.decision_function(valid_x)


#
train_lines = open('afew2_train_filelist.txt').readlines()
test_lines = open('afew2_test_filelist.txt').readlines()
valid_lines = open('afew2_valid_filelist.txt').readlines()
#
mat_train = numpy.zeros((len(train_lines), 7))
mat_valid = numpy.zeros((len(valid_lines), 7))
mat_test = numpy.zeros((len(test_lines), 7))
#
mat_train_feats = numpy.zeros((len(train_lines), 1440))
mat_valid_feats = numpy.zeros((len(valid_lines), 1440))
mat_test_feats = numpy.zeros((len(test_lines), 1440))

mat_train_pred = numpy.zeros((len(train_lines),1))
mat_valid_pred = numpy.zeros((len(valid_lines),1))
mat_test_pred = numpy.zeros((len(test_lines),1))

#
def get_id_from_clip(t, clip_idx):
    for i in xrange(len(eval(t + '_x'))):
        elem = os.path.split(eval(t + '_path')[i])[-1]
        if 'org' not in elem or 'flip' in elem:
            continue

        if int(elem.split('_')[4]) == clip_idx:
            return i
    return -1    

for l in train_lines:
    idx, the_id = l.split()
    item_idx = get_id_from_clip('train', int(the_id))
    if item_idx < 0: continue
    mat_train_feats[int(idx),:] = train_x[item_idx]
    mat_train[int(idx)] = emotion_train[item_idx]
    mat_train_pred[int(idx)] = pred_train[item_idx]

for l in test_lines:
    idx, the_id = l.split()
    item_idx = get_id_from_clip('test', int(the_id))
    if item_idx < 0: continue
    mat_test_feats[int(idx),:] = test_x[item_idx]
    mat_test[int(idx)] = emotion_test[item_idx]
    mat_test_pred[int(idx)] = pred_test[item_idx]

for l in valid_lines:
    idx, the_id = l.split()
    item_idx = get_id_from_clip('valid', int(the_id))
    if item_idx < 0: continue
    mat_valid_feats[int(idx),:] = valid_x[item_idx]
    mat_valid[int(idx)] = emotion_valid[item_idx]
    mat_valid_pred[int(idx)] = pred_valid[item_idx]

#
#for l in test_lines:
#    idx, the_id = l.split()
#    (x,y) = test[int(the_id)]   
#    prediction = clf.predict(x)
#    mat_test_feats[int(idx),:] = x[:]
#    prediction_1h = numpy.zeros((1, 7))
#    prediction_1h[prediction] = 1.
#    mat_test[int(idx)] = prediction_1h
#
#for l in valid_lines:
#    idx, the_id = l.split()
#    (x,y) = valid[int(the_id)]   
#    prediction = clf.predict(x)
#    mat_valid_feats[int(idx),:] = x[:]
#    prediction_1h = numpy.zeros((1, 7))
#    prediction_1h[prediction] = 1.
#    mat_valid[int(idx)] = prediction_1h
#
numpy.save('/Tmp/aggarwal/AJM_train_feats.npy', mat_train_feats)
numpy.save('/Tmp/aggarwal/AJM_test_feats.npy', mat_test_feats)
numpy.save('/Tmp/aggarwal/AJM_valid_feats.npy', mat_valid_feats)

numpy.save('/Tmp/aggarwal/AJM_train_preds.npy', mat_train_pred)
numpy.save('/Tmp/aggarwal/AJM_test_preds.npy', mat_test_pred)
numpy.save('/Tmp/aggarwal/AJM_valid_preds.npy', mat_valid_pred)

numpy.save('/Tmp/aggarwal/AJM_learned_on_othersonly_predict_on_test_scores.npy', mat_test)
numpy.save('/Tmp/aggarwal/AJM_learned_on_othersonly_predict_on_train_scores.npy', mat_train)
numpy.save('/Tmp/aggarwal/AJM_learned_on_othersonly_predict_on_valid_scores.npy', mat_valid)


