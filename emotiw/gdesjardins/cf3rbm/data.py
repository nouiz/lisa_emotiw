import glob
import numpy
from PIL import Image
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load

BASEDIR = '/data/lisa/data/faces/EmotiW/Aligned_Images/alignedAvgd/'
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preproc_small_grey(x):
    x = x.convert('L')
    x.thumbnail((48,48), Image.ANTIALIAS)
    return x


class Category(object):

    def __init__(self, base, label):
        self.base = base
        self.label = label
        clip_base = '%s/%s' % (base, label)
        fnames = glob.glob('%s/*-001.png' % clip_base)
        fnames.sort()
        
        self.num_clips = len(fnames)
        self.clips = []
        for _fname in fnames:
            fname = _fname.split('/')[-1].split('-')[0]
            self.clips += [Clip(clip_base, fname)]


class Clip(object):

    def __init__(self, base, clipid):
        self.base = base
        self.clipid = clipid
        self.frames = glob.glob('%s/%s-*.png' % (base, clipid))
        self.frames.sort()
        # will contain pointers to rows of dataset.X
        self.frames_index = []
        self.num_frames = len(self.frames)

    def get_frame(self, i, preproc=None):
        assert i < len(self.frames)
        return Image.open(self.frames[i])

    def get_random_frame(self, rng):
        i = rng.random_integers(low=0, high=len(self.frames)-1)
        return self.get_frame(i)

def load_all_frames(which_set):
    assert which_set in ['Train', 'Val']

    category = {}
    num_frames = 0
    for label in LABELS:
        base = '%s/%s' % (BASEDIR, which_set)
        category[label] = Category(base, label)
        for clip in category[label].clips:
            num_frames += clip.num_frames

    X = numpy.zeros((num_frames, 48*48))
    y = numpy.zeros((num_frames))
    idx = 0
    for labid, label in enumerate(LABELS):
        print 'Loading category %s' % label
        base = '%s/%s' % (BASEDIR, which_set)
        category[label] = Category(base, label)
        for i, clip in enumerate(category[label].clips):
            for j, frame in enumerate(clip.frames):
                print 'Loading clip %i/%i, frame %i/%i \r' %\
                      (i, category[label].num_clips, j, clip.num_frames),
                img = preproc_small_grey(Image.open(frame))
                X[idx, :] = numpy.array(img).flatten()
                y[idx] = labid
                clip.frames_index += [idx]
                idx += 1
        print '\n'

    return (X, y), category


class EmotiwFaces(dense_design_matrix.DenseDesignMatrix):
    """
    Pylearn2 wrapper for the MNIST-Plus dataset.
    """

    def __init__(self, which_set, center = False, gcn=False,
                 one_hot=False, seed=132987):
        
        assert which_set in ['Train', 'Val']
        self.rng = numpy.random.RandomState(seed)
        self.which_set = which_set
        self.center = center
        self.gcn = gcn
        self.one_hot = one_hot

        (X, y), self.meta = load_all_frames(which_set)
        ## filter out pure-black images ###
        X = (X / 255).astype(config.floatX)
        y = y.astype(config.floatX)

        if gcn:
            goodidx = numpy.where(numpy.sum(X, axis=1) != 0)
            meanx = numpy.mean(X, axis=1)[:,None]
            stdx  = numpy.std(X, axis=1)[:,None]
            X[goodidx] = (X[goodidx] - meanx[goodidx]) / stdx[goodidx]

        if center:
            X -= numpy.mean(X, axis=0)

        if one_hot:
            one_hot = numpy.zeros((y.shape[0],7),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot

        view_converter = dense_design_matrix.DefaultViewConverter((48,48,1))
        super(EmotiwFaces, self).__init__(X = X, y =y,
                view_converter = view_converter)

    def get_random_frame_batch(self, batch_size=128):
        idx = self.rng.random_integers(low=0, high=len(self.X)-1, size=batch_size)
        return (self.X[idx], self.y[idx])

    def get_random_framepair_batch(self, batch_size=128):
        bX = numpy.zeros((batch_size, 2*self.X.shape[1]))
        if self.one_hot:
            by = numpy.zeros((batch_size, self.y.shape[1]))
        else:
            by = numpy.zeros(batch_size)

        cat_indices = self.rng.random_integers(low=0, high=len(LABELS), size=batch_size)

        k = 0
        for k in xrange(batch_size):
            cat_idx = self.rng.random_integers(low=0, high=len(LABELS)-1, size=1)
            cat = self.meta[LABELS[cat_idx]]
            clip_idx = self.rng.random_integers(low=0, high=cat.num_clips-1, size=1)
            clip = cat.clips[clip_idx]
            rel_frame1_idx = self.rng.random_integers(low=0, high=clip.num_frames-2, size=1)
            rel_frame2_idx = self.rng.random_integers(low=rel_frame1_idx, high=clip.num_frames-1, size=1)
            abs_frame1_idx = clip.frames_index[rel_frame1_idx]
            abs_frame2_idx = clip.frames_index[rel_frame2_idx]
            #print 'index %i: category (%i, %s), clipid (%i, %s), frame1 %i, frame2 %i' %\
                   #(k, cat_idx, LABELS[cat_idx], clip_idx, clip.clipid, rel_frame1_idx, rel_frame2_idx)
            bX[k] = numpy.hstack((self.X[abs_frame1_idx], self.X[abs_frame2_idx]))
            by[k] = self.y[abs_frame1_idx]

        return (bX, by)
