from emotiw.common.datasets.faces.imageseq import ImageSequenceDataset
from emotiw.common.datasets.faces.faceimages import FaceImagesDataset
from emotiw.common.datasets.faces.faceimages import basic_7emotion_names
import numpy
import glob

class EmotiwPreprocDataset(ImageSequenceDataset):
    def __init__(self, emotion, size = (96, 96), num_channels = 1, img_per_seq = 3):
        if isinstance(emotion, str):
            emotion = basic_7emotion_names.index(emotion.lower())

        sets = glob.glob('/data/lisa/data/faces/EmotiW/preproc/static/*/*_' + str(emotion) + '.npy')
        sets.sort()

        self.images = []
        for s in sets:
            img = numpy.memmap(s, mode='r', dtype='float32')
            img = img.view()
            img.shape = (len(img)/(size[0]*size[1]*num_channels), size[0], size[1], num_channels)
            self.images.append(img)

        self.emotion = emotion
        self.ips = img_per_seq

    def get_sequence(self, idx):
        lgts = [len(x) for x in self.images]
        sums = [sum(lgts[:i+1]) for i in xrange(len(lgts))]
        for i, s in enumerate(sums):
            if s > idx:
                return StaticPreprocSequence([self.images[i][idx - sum(lgts[:i+1])]]*self.ips, self.emotion)

    def __len__(self):
        return sum([len(x) for x in self.images])


class StaticPreprocSequence(FaceImagesDataset):
    def __init__(self, images, emotion):
        self.images = images
        self.emotion = emotion

    def get_7emotion_index(self, idx):
        return self.emotion

    def get_original_image(self, idx):
        return self.images[idx]

    def get_original_image_path(self, idx):
        raise NotImplementedError('Not available')

    def __len__(self):
        return len(self.images)
