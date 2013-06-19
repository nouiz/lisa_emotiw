import numpy as np
import emotiw.common.datasets.faces.afew2 as afew2
from collections import defaultdict
import sys

import matplotlib.pyplot as plt
from pylearn2.gui import patch_viewer

d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)
classes = d.emotionNames.values()

#from PIL import Image, ImageSequence
#im = Image.open('moving_cat.gif')
#frames = [np.array(frame.copy()) for frame in ImageSequence.Iterator(im)]
#print frames
#plt.imshow(frames)
#plt.show()

def get_facetube(s,i):
    print s,i,'then',
    if s=='Val':
        i += sum([info[0]=='Train' for info in d.seq_info])

    print i
    x = d.get_facetubes(i)
    y = d.emotionNames.values().index(d.labels[i])
    if len(x)==0:
        return (None,None)

    idx = np.random.random_integers(0,len(x)-1)
    return (x[idx].astype('float32'),y)

def show_facetube(x):
    # implement saving from show_examples
    examples = x
    rows = int(np.sqrt(examples.shape[0]))
    cols = int(examples.shape[0]/rows + 1)

    examples /= np.abs(examples).max()

    if examples.shape[3] == 1:
        is_color = False
    elif examples.shape[3] == 3:
        is_color = True
    else:
        print examples.shape[3],"?"

    #take 36 images
    if rows >= 6:
        rows = 6
        cols = 6
        idx = np.arange(0.0,examples.shape[0],examples.shape[0]/(0.+36))
        idx = [int(id) for id in idx]
    else:
        idx = np.arange(examples.shape[0])

    pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color=is_color)
    for i in idx:
        pv.add_patch(examples[i,:,:,:], activation = 0.0, rescale = False)

    pv.show()

def len_facetubes(s):
    return sum([info[0]==s for info in d.seq_info])


def len_shuffle(s):
    return len(d[s]['X'])

def get_shuffled(s,i):
    return (d[s]['X'][i],d[s]['y'][i])

def show_image(x):
    plt.imshow(x)
    plt.show()
    pass

test_type = sys.argv[1]
if test_type=="facetube":
    get_data = get_facetube
    show_sample = show_facetube
    len_data = len_facetubes

elif test_type=="shuffled":
    get_data = get_shuffled
    show_sample = show_image
    len_data = len_shuffle
    d = {}
    for s in ['Train','Val']:
        d[s] = {'X': np.load('%s_X.npy' % s),
                'y': np.load('%s_y.npy' % s)}

else:
    raise ValueError("test_type is invalid: "+test_type)

results = defaultdict(list)

for s in ['Val']:#['Train','Val']:
    idx = np.arange(len_data(s))
    np.random.shuffle(idx)
    for i in range(50):
        print list(enumerate(classes))
        (x, y) = get_data(s,idx[i])
        if x is None:
            continue
        show_sample(x)
        c = input("which emotion? ")
        results[s].append(c==y)
        print "it was", classes[y]

    print np.mean(results[s])*100.0,"% accuracy"
