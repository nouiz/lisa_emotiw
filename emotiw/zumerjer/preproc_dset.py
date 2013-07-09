from emotiw.common.datasets.faces.imageseq import ImageSequenceDataset
from emotiw.common.datasets.faces.faceimages import FaceImagesDataset
import numpy
import os


class EmotiwPreprocSeq(FaceImagesDataset):
    def __init__(self, images, emotion):
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
    def __init__(self, emotion):
        abs_path = '/data/lisatmp2/emotiw/dataset'
        image_path = os.path.join(abs_path, emotion + '_x.npy')
        emote_path = os.path.join(abs_path, emotion + '_y.npy')

        self.emotions = numpy.memmap(emote_path, mode='c', dtype=numpy.uint8)
        self.images = numpy.memmap(image_path, mode='c', dtype=numpy.uint8)
        self.images = self.images.view()
        self.images.shape = (len(self.emotions), 3, 48, 48, 3)

    def get_sequence(self, i):
        return EmotiwPreprocSeq(self.images[i], self.emotions[i])

    def get_7emotion_index(self, i):
        return self.emotions[i]

    def __len__(self):
        return len(self.images)
