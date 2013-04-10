import glob
import os
import csv
from imageseq import ImageSequenceDataset
from faceimages import FaceImagesDataset

# Subclasses of FaceImagesDataset


class ImageSequence(FaceImagesDataset):
    def __init__(self, dataset_name, relative_image_base_directory, image_glob, relative_bbox_directory, emotionName):
        super(ImageSequence, self).__init__(dataset_name, relative_image_base_directory)

        self.bbox = []
        self.imageRelativePath = []  # Relative path to images
        self.emotionIndex = FaceImagesDataset.base_7emotion_names.index(emotionName.lower())
        self.imageIndex = {}

        # Fetch images
        imagesName = glob.glob(os.path.join(relative_image_base_directory, image_glob))

        # Sort the frames
        imagesName.sort()

        # Builds the data
        idx = 0
        for img in imagesName:
            self.imageRelativePath.append(img)
            self.imageIndex[img] = idx
            bbxName = os.path.basename(img).replace(".jpg", ".txt")
            bboxList = []
            bbxName = os.path.join(relative_bbox_directory, bbxName)
            if os.path.exists(bbxName):
                with open(bbxName) as f:
                    reader = csv.reader(f, delimiter=' ')
                    for row in reader:
                        bboxList.append([float(x) for x in row[:4]])
            else:
                bboxList = None

            self.bbox.append(bboxList)
            idx += 1

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]

    def __len__(self):
        return len(self.imageRelativePath)

    def get_original_image_path_relative_to_base_directory(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.imageRelativePath[i]
        else:
            return None

    def get_7emotion_index(self, i):
        return self.emotionIndex

    def get_bbox(self, i):
        return self.get_picasa_bbox(i)

    def get_picasa_bbox(self, i):
        return self.bbox[i]


class AFEWImageSequenceDataset(ImageSequenceDataset):
    def __init__(self):
        self.absolute_base_directory = "/data/lisa/data/faces/AFEW/images/"
        self.imagesequences = []
        self.labels = []
        self.trainIndexes = []
        self.validIndexes = []
        self.emotionNames = {"Angry": "anger", "Disgust": "disgust", "Fear": "fear", "Happy": "happy",
                             "Neutral": "neutral", "Sad": "sad", "Surprise": "surprise"}
        # For each emotion subfolder
        idx = 0
        directories = os.listdir(self.absolute_base_directory)
        directories.sort()
        for emotionName in directories:
            # Check if it is a emotion subfolder
            emotionDir = os.path.join(self.absolute_base_directory, emotionName)
            if not os.path.isdir(emotionDir) or emotionName not in self.emotionNames.keys():
                continue
            picassaBboxDir = os.path.join("/data/lisa/data/faces/AFEW/picasa_boxes/", emotionName)

            # Find all images
            fileNames = glob.glob(os.path.join(emotionDir, "*.jpg"))

            # Find all unique sequences
            uniqueSequence = list(set([name.split("_")[0] for name in fileNames]))
            uniqueSequence.sort()

            # For each unique sequence
            for sequence in uniqueSequence:
                # Load the Image Sequence object
                seq = ImageSequence("AFEW", emotionDir, "{0}_*.jpg".format(sequence), picassaBboxDir,
                                    self.emotionNames[emotionName])
                self.imagesequences.append(seq)
                # Save label
                self.labels.append(self.emotionNames[emotionName])
                # Save if in train or valid
                self.trainIndexes.append(idx)

                idx += 1

        return

    def __len__(self):
        return len(self.imagesequences)

    def get_sequence(self, i):
        return self.imagesequences[i]

    def get_label(self, i):
        return self.labels[i]

    def get_standard_train_test_splits(self):
        # Only one fold
        return [(self.trainIndexes, self.validIndexes)]




    
    

