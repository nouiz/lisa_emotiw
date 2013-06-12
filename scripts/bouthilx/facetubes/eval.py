import emotiw.common.datasets.faces.afew2 as afew2
from collections import defaultdict
from theano import tensor as T
from theano import function
from pylearn2.utils import serial

try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)

X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)
#y = T.argmax(Y, axis=1)
f = function([X], y)

pools = defaultdict(list)
targets = defaultdict(list)
for clip, info, target in zip(d.imagesequences,d.seq_info,d.labels):
    faces = []
    predictions = []
    for facetube in d.get_facetubes(i)
        faces += [face for face in facetube]
    for face in faces:
        predictions.append(f(face))

    # max class of mean on face-tubes
    pools[info[0]].append(np.argmax(np.array(predictions).mean(0)))
    targets[info[0]].append(target)

for name, vals in pools.items():
    print name, (np.array(vals)==np.array(targets[name])).mean()
