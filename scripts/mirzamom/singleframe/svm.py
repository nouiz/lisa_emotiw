from sklearn import svm
import numpy
from pylearn2.utils import serial
from sklearn import preprocessing
import ipdb

train = serial.load('train.pkl')
valid = serial.load('valid.pkl')


train_x, train_y = train['x'], train['y']
valid_x, valid_y = valid['x'], valid['y']

#train_X = preprocessing.scale(train_x)
#valid_x = preprocessing.scale(valid_x)

c_vals = numpy.logspace(-5, 10, num = 15)
for c in c_vals:
    clf = svm.LinearSVC(C = c)
    #clf = svm.SVC(C = c)
    clf.fit(train_x, train_y)
    print c, clf.score(valid_x, valid_y)
