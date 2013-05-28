# Base class for NCKU-like datasets.
# Original code by abhi (abhiggarwal@gmail.com)

import os
import numpy as np
import sys
import glob
from scipy import io as sio
import unicodedata
import math
import json

from emotiw.common.utils.pathutils import locate_data_path
from faceimages import FaceImagesDataset

class NckuBasedDataset(FaceImagesDataset):
    def __init__(self, name, path):
        super(NckuBasedDataset, self).__init__(name, path)
        self.rel_dir = os.path.split(path)[1]
        
    def read_json_keypoints(self):
        self.keyPoints = []
        self.bbox = []
        translation_dict = {'eye_left': 'right_eye_pupil', 'eye_right': 'left_eye_pupil', 'center': 'face_center',
                            'mouth_right': 'mouth_left_corner', 'mouth_left': 'mouth_right_corner', 
                            'mouth_center': 'mouth_center', 'nose': 'nose_tip'}
        #print 'length ofimageIndex'
        #print len(self.imageIndex)
        for file in self.imageIndex :
             relPath = self.get_original_image_path_relative_to_base_directory(self.imageIndex[file])
             pathJson = os.path.join(self.absolute_base_directory, '..', 'mashapeKpts', self.rel_dir, relPath)
             pathJson = os.path.splitext(pathJson)[0]

             if 'IHDPHeadPose' in self.absolute_base_directory and '_flip' in relPath: 
                 pathJson = pathJson[:-1] #files are named incorrectly: _flip.jpg -> fli.json
                #Really should be moved to IHDPHeadPose.py
             pathJson = pathJson + '.json'

             bbox = ()
             keyDict = {}
             try:
                 try:
                     jsonData = open(pathJson)
                     data = json.load(jsonData)
                     if len(data) == 0:
                         self.keyPoints.append({})
                         self.bbox.append(())
                     else:
                         bbox = ()
                         keydict = {}
                         self.out += 1
                         #print self.out
                         for key in data[0]:
                             if key not in ['confidence', 'tid', 'attributes', 'height', 'width', 'center'] :
                                 keyDict[translation_dict[str(key)]] = (data[0][key]['x'],data[0][key]['y'])
                                 #print keyDict[key]
                         bbox = (data[0]['center']['x'], data[0]['center']['y'], data[0]['width'], data[0]['height'])
                 except IOError:
                    keyDict = {}   
                    bbox = []
             finally:
                self.keyPoints.append(keyDict)
                self.bbox.append(bbox)

    def get_keypoints_location(self, i):
        if i >= 0 and i < len(self.keyPoints):
            return self.keyPoints[i]

    def __len__(self):
        return len(self.images)

    def get_pan_tilt_and_roll(self, i):
        if i >= 0 and i < len(self.images):
            return (self.pan[i], self.tilt[i], self.roll[i])
        else:
            return None 

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.relPaths[i]

    def get_subject_id_of_ith_face(self, i):
        return None

    def get_head_pose(self, i):
        return None

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]
    
    def get_original_bbox(self, i):
        #bounding box in [x0, y0, x1, y1] format.
        try:
            x0, y0, w, h = self.bbox[i]
            return [x0 - w/2, y0 - h/2, x0 + w/2, y0 + h/2]
        except ValueError:
            return None

    def get_eyes_location(self, i):
        try:
            return [self.keyPoints[i]['right_eye_pupil'][0], self.keyPoints[i]['right_eye_pupil'][1],
                    self.keyPoints[i]['left_eye_pupil'][0], self.keyPoints[i]['left_eye_pupil'][1]]
        except KeyError:
            return None
