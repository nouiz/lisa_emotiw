from emotiw.common.datasets.faces.imageseq import ImageSequenceDataset
from emotiw.common.datasets.faces.faceimages import FaceImagesDataset, basic_7emotion_names
import numpy
import os
import glob


class EmotiwPreprocSeq(FaceImagesDataset):
    def __init__(self, images, emotion, size, num_channels):
        self.images = images
        self.emotion = emotion

    def get_7emotion_index(self, i):
        return self.emotion

    def get_original_image_path(self, i):
        raise NotImplementedError('Not available')

    def get_original_image(self, i):
        return self.images[i]

    def __len__(self):
        return len(self.images)


class EmotiwPreprocDataset(ImageSequenceDataset):
    def __init__(self, emotion, size = (48, 48), num_channels = 3, img_per_seq = 3):
        if isinstance(emotion, str):
            self.emotion = basic_7emotion_names.index(emotion)
            emotion = emotion[0].upper() + emotion[1:]
        else:
            self.emotion = emotion
            emotion = basic_7emotion_names[emotion]

        files = glob.glob('/data/lisa/data/faces/EmotiW/preproc/seq/*/'+emotion+'/*.npy')
        files.sort()

        self.seq = []
        self.lgts = []
        for f in files:
            seq = numpy.memmap(f, mode='r', dtype='float32')
            lgt = (len(seq)/(size[0]*size[1]*num_channels))
            seq.shape = (lgt, size[0], size[1], num_channels)
            self.seq.append(seq)

        self.ips = img_per_seq
        self.size = size
        self.num_channels = num_channels

    def get_sequence(self, i):
        return EmotiwPreprocSeq(self.seq[i], self.emotion, self.size, self.num_channels)

    def get_7emotion_index(self, i):
        return self.emotion

    def __len__(self):
        return len(self.seq)
