from imageseq import ImageSequenceDataset
from faceimages import FaceImagesDataset, basic_7emotion_names
import re
import glob
import os


class AttousaFrame(FaceImagesDataset):
    def __init__(self, clip_path, emotion):
        self.emotion = emotion
        self.images = glob.glob(os.path.join(clip_path, '*.png'))
        self.images.sort()

        super(AttousaFrame, self).__init__('Attousa Frame Data')

    def __len__(self):
        return len(self.images)

    def get_7emotion_index(self, idx):
        return [x.lower() for x in basic_7emotion_names].index(self.emotion.lower())

    def get_keypoints_location(self, i):
        raise NotImplementedError('No keypoints available for this dset.')

    def get_original_image_path(self, i):
        return self.images[i]


class AttousaDataset(ImageSequenceDataset):
    def __init__(self):
        self.sequences = []

        abspath = '/data/lisatmp2/emo_video/new_clips/ExtractedFrames/'
        emotion_re = re.compile('[a-zA-Z]*(?=[0-9]*)')

        for folder in os.listdir(abspath):
            emotion = re.match(emotion_re, folder).group(0).replace("noclass", "neutral")
            self.sequences.append(AttousaFrame(os.path.join(abspath, folder), emotion))

        super(AttousaDataset, self).__init__('Attousa Extracted Frames')

    def __len__(self):
        return len(self.sequences)

    def get_sequence(self, i):
        return self.sequences[i]

    def get_label(self, i):
        return self.sequences[i].emotion
