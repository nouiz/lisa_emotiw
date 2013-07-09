from emotiw.common.datasets.imageseq import ImageSequenceDataset
from emotiw.common.datasets.faceimages import FaceImagesDataset
import numpy

class EmotiwPreprocDataset(ImageSequenceDataset):
    def __init__(self, emotion, size = (48, 48), num_channels = 3, img_per_seq = 3):
        self.images = numpy.memmap('/data/lisatmp2/emotiw/dataset_' + emotion + '_x.npy')        
        self.emotions = numpy.memmap('/data/lisatmp2/emotiw/dataset_' + emotion + '_y.npy')        
        self.images = self.images.view()
        self.images.shape = (len(self.emotions), size[0], size[1], num_channels)
        self.ips = img_per_seq

    def get_sequence(self, idx):
        return StaticPreprocSequence([self.images[idx]]*self.ips, self.emotions(idx))

    def __len__(self):
        return len(self.emotions)

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
