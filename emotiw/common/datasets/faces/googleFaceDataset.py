import os
import numpy as np
import sys
import glob
import cPickle
from scipy import io as sio
import unicodedata

from faceimages import FaceImagesDataset

# Subclasses of FaceImagesDataset
class GoogleFaceDataset(FaceImagesDataset):
    
    def __init__(self):
        super(GoogleFaceDataset,self).__init__("GFD", "faces/GoogleDataset/")
        
        # Load the dataset's pickle file
        data = cPickle.load(open(self.absolute_base_directory+"Clean/latest.pkl","rb"))
        data = data[1:]
        
        self.labels = data[0]
        self.ids = data[2]
        self.queryIds = data[3]
        self.tags = data[4]
        self.tagNames = data[7]
        self.X1 = data[8]
        self.Y1 = data[9]
        self.X2 = data[10]
        self.Y2 = data[11]
    
    def get_name(self):
        return "GoogleFaceDataset"
        
    def __len__(self):
        return len(self.labels)

    def get_original_image_path_relative_to_base_directory(self, i):
        return "images/" + str(self.queryIds[i]) + "/" + str(self.ids[i]) + ".png"
        
    def get_index_from_image_filename(self, imgFileName):
        filename = imgFileName.split("/")[-1]
        imageId = int(filename[:-4])
        return (self.ids == imageId).argmax()

    def get_original_bbox(self, i):
        return [[self.X1[i], self.Y1[i],
                 self.X2[i] - self.X1[i],
                 self.Y2[i] - self.Y1[i]]]

    def get_7emotion_index(self, i):
        return self.labels[i].argmax()
        
    def get_detailed_emotion_label(self, i):
        return self.tagNames[self.tags[i].argmax()]