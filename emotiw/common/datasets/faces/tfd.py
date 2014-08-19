# Copyright (c) 2013 University of Montreal, Pascal Vincent
# Pierre-Luc Carrier, Vincent Archambault, Hani Almousli
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of the authors and contributors to this software may not be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import sys
import glob
from scipy import io as sio
import unicodedata

from emotiw.common.utils.pathutils import locate_data_path
from faceimages import FaceImagesDataset

def get_dataset_constructors():
    return { "TorontoFaceDataset": TorontoFaceDataset,
             "StaticCKPlus": StaticCKPlus,
             "SFEW": SFEW,
             "MSFDE": MSFDE,
             "IndianFaceDatabase": IndianFaceDatabase,
             "Jaffe": Jaffe,
             "Jacfee": Jacfee,
             "ArFace": ArFace,
             "Kdef": Kdef,
             "NimStim": NimStim,
             "Pofa": Pofa }

# Subclasses of FaceImagesDataset

class TorontoFaceDataset(FaceImagesDataset):
    """
    The Toronto Face Dataset (TFD) was set up by Josh Susskind as a union of several face datasets
    It has both a labeled examples (with 7 basic emotion labels) and unlabeled examples.
    
    See the techreport here:
    http://aclab.ca/users/josh/TFD.html
    
    Here are the original links to download the data:
    http://www.cs.toronto.edu/~jsusskin/TFD/TFD_48x48.mat
    http://www.cs.toronto.edu/~jsusskin/TFD/TFD_96x96.mat
    http://www.cs.toronto.edu/~jsusskin/TFD/TFD_info.mat
    
    For each original dataset we indicate (when we know them):
    usual abbreviated name,
    full name,
    abbreviaiton in TFD mat file or TFD tech report table 1,
    path relative to data directory (/data/lisa/data/)

    The following should have emotion labels and FACS info:
    - JACFEE: Japanese and Caucasian Facial Expressions of Emotion ('jacfee') faces/JACFEE/standard_expressor_set/
    - MMI: MMI Facial Expression Database ('MMI') faces/mmifacedb/
    - MSFDE: Montreal Set of Facial Displays of Emotion ('msfde')   faces/MSFDE/MSFDE/MSFDE/
    - CK+: The Extended Kohn Kanade Dataset ('dfat')  faces/cmu_expression/CK+/
    
    The following have emotions but no FACS:
    - IndianFace: The Indian Face Database ('indian') faces/IndianFaceDatabase
    - JAFFE: The Japanese Female Facial Expression Database ('jaffe') faces/jaffe 
    - POFA: Pictures of Facial Affect ('pofa') faces/POFA/POFA/
    - KDEF: The Karolinska Directed Emotional Faces ('kdef') faces/KDEF_and_AKDEF 
    - NimStim: The NimStim face stimulus set ('nim') faces/NimStim

    Couldn't find the source for downloading:
    'inaoe'  Cruz, C., Sucar, L., & Morales, E. (2008). Real-time face recognition for human-robot interaction. Paper presented at the IEEE International Conference on Automatic Face and Gesture Recognition.
    'cafe'   California Facial Expressions (CAFE) (Dailey, Cottrell, & Reilly, 2001)
    
    Couldn't associate the abbreviation in the mat file and list of databases in the TFD tech report Table 1:
    'fotosearch'
    'aging'
    'cad'
    'josh'
    'livegoogemo'
    'relived'
    'spontaneous'
    """
    
    def __init__(self):
        super(TorontoFaceDataset, self).__init__("TFD", "faces/TFD/")
        # Load original 48x48 images
        tfd_48x48 = sio.loadmat(os.path.join(self.absolute_base_directory, "TFD_48x48.mat"))
        self.images = tfd_48x48["images"]
        self.labels = tfd_48x48['labs_ex']
        self.folds = tfd_48x48['folds']

        # Load mapping file to original dataset
        self.mapping = sio.loadmat(locate_data_path("faces/TFD_extra/TFD_info.mat"))
        self.mapping = [x[0].encode('ascii', 'ignore') for x in self.mapping["tfd_info"]["imfiles"][0, 0].flatten()]
        self.knownDataset = {}

    def __len__(self):
        return self.images.shape[0]

    def get_7emotion_index(self, idx):
        label_idx = self.labels[idx][0] - 1
        if label_idx < 0:
            return None
        return label_idx

    def get_original_image(self, idx):
        return self.images[idx]

    def get_keypoints_location(self, idx):
        return [{'right_eye_inner_corner': (15, 10), 
                 'right_eye_outer_corner': (5, 10),
                 'left_eye_inner_corner': (32, 10),
                 'left_eye_outer_corner': (43, 10),
                 'nose_tip': (24, 24), 
                 'mouth_center': (24, 39), 
                 'mouth_right_corner': (14, 36), 
                 'mouth_left_corner': (34, 36)}]

    def get_original_image_path_relative_to_base_directory(self, i):
        return None

    def get_original_image_path(self, i):
        return None

    def get_source_dataset_position(self, idx, datasetDict=None):
        """
        Returns tuple (string, int) -> (Name of dataset, position image dans le dataset d'origine)

        datasetDict : A dictionary containing FaceImagesDataset subclasses. This avoids reloading the
                      dataset again and again in order to find the index number.
                      If not provided, this function will cache the dataset in the class variable self.knownDataset
        """
        fileName = self.mapping[idx]

        # Figure out which dataset it is
        datasetName = self._extractDatasetName(fileName)
        if datasetName is None:
            return None

        # Reformat file name to fit the original name
        originalFileName = self._getOriginalFileName(datasetName, fileName)
        if originalFileName is None:
            return None

        # Load the dataset and find the position of the image
        if datasetDict is not None and datasetName in datasetDict.keys():
            dSet = self.knownDataset[datasetName]
        elif datasetName in self.knownDataset.keys():
            dSet = self.knownDataset[datasetName]
        else:
            dSet = self._loadDataset(datasetName)
        if dSet is None:
            return None

        return datasetName, dSet.get_index_from_image_filename(originalFileName)

    def _extractDatasetName(self, filename):
        if filename[0:4] == "pofa":
            return "Pofa"
        if filename[0:3] == "nim":
            return "NimStim"
        if filename[0:4] == "kdef":
            return "Kdef"
        if filename[0:5] == "jaffe":
            return "Jaffe"
        if filename[0:6] == "indian":
            return "IndianFaceDatabase"
        if filename[0:5] == "msfde":
            return "MSFDE"
        else:
            return None

    def _getOriginalFileName(self, datasetName, fileName):
        if datasetName == "Pofa":
            name = fileName.split("_")[2]
            return name.replace(".jpg", ".TIF")
        if datasetName == "NimStim":
            name = "_".join(fileName.split("_")[4:])
            return name.replace(".jpg", ".bmp")
        if datasetName == "Kdef":
            name = fileName.split("_")[1]
            return name.replace(".jpg", ".JPG")
        if datasetName == "Jaffe":
            name = fileName.split("_")[1]
            return name.replace(".jpg", ".tiff")
        if datasetName == "IndianFaceDatabase":
            sex = "males" if fileName.split("_")[2] == "mal" else "females"
            subjectId = fileName.split("_")[3]
            imgName = fileName.split("_")[1] + ".jpg"
            return os.path.join(sex, str(subjectId), imgName)
        if datasetName == "MSFDE":
            name = fileName.split("_")[1]
            return name.replace(".jpg", ".tif")
        else:
            return None

    def _loadDataset(self, datasetName):
        if datasetName == "Pofa":
            return Pofa()
        if datasetName == "NimStim":
            return NimStim()
        if datasetName == "Kdef":
            return Kdef()
        if datasetName == "Jaffe":
            return Jaffe()
        if datasetName == "IndianFaceDatabase":
            return IndianFaceDatabase()
        if datasetName == "MSFDE":
            return MSFDE()
        else:
            return None


class StaticCKPlus(FaceImagesDataset):
    """Extended Cohn Kanade (CK+) with only all first (neutral) and last (emotiv) images of the sequences""" 
    def __init__(self):

        # Call parent constructor
        super(StaticCKPlus,self).__init__("CK+", "faces/cmu_expression/CK+/")

        emotionsDic = { "anger":0, "disgust":1, "fear":2 , "happy":3, "sadness":4, "surprise":5, "neutral":6, "contempt":7}

        import zipfile
        z = zipfile.ZipFile(self.absolute_base_directory+'Landmarks.zip', 'r')

        path = self.absolute_base_directory+"cohn-kanade-images/"
        emotionFile = self.absolute_base_directory + "emotions.txt"

        lstEmotions =[]
        lstpaths = []
        lstsubject = []
        self.lstImages=[]
        f = open(emotionFile)
        for line in f:
            v = line.strip().split(',')
            prefix ='S'
            i=0
            while i < 4-len(v[0])-1:
                prefix +='0'
                i+=1

            lstpaths.append("cohn-kanade-images/"+prefix+v[0]+'/'+v[1])
            lstEmotions.append(emotionsDic[v[2]])
            lstsubject.append((int)(v[0]))

        self.csubject = 0 # Subject Counts
        for p in zip(lstpaths,lstEmotions,lstsubject):
            imp,emo, sub = zip(p)

            files = os.listdir(self.absolute_base_directory+imp[0])
            np.sort(files)            
            
            #Non Neutral
            self.csubject +=1         
            
            keypoints = self.read_keypoints('Landmarks'+p[0][18:len(p[0])]+"/"+files[len(files)-1][0:len(files[len(files)-1])-4]+'_landmarks.txt',z)
            self.lstImages.append([ p[0]+"/"+files[len(files)-1] ,emo[0],sub[0], keypoints])            
            #import pdb;pdb.set_trace()
            #Neutral:
            keypoints = self.read_keypoints('Landmarks'+p[0][18:len(p[0])]+"/"+files[len(files)-1][0:len(files[len(files)-1])-4]+'_landmarks.txt',z)
            self.lstImages.append([ p[0]+"/"+files[1] ,6,sub[0],keypoints])
            #import pdb;pdb.set_trace()

    def read_keypoints(self,path,z):
        featurePoints=[]
        #import pdb;pdb.set_trace()
        try:
            lstpoints = z.read(path).split('\n')
            c = 0
            while c<len(lstpoints):
                point = lstpoints[c]
                if point != '':
                    featurePoints.append(point.split())
                c+=1
        except:
            featurePoints=[]
        #import pdb;pdb.set_trace()    
        return featurePoints

    def __len__(self):
        return len(self.lstImages)

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.lstImages[i][0]

    def get_n_subjects(self):
        return self.csubject

    def get_7emotion_index(self, i):
        index = self.lstImages[i][1]
        if index < 7:
            return index
        return None

    def get_id_of_kth_subject(self, k):
        return str(k)

    def get_subject_id_of_ith_face(self, i):
        return str(self.lstImages[i][2])
    
    def get_keypoints_list(self,i):
        """
        This will return a bunch of points. to know what each index means, refere to an image called keypoints inside the ck+ directory        
        """
        return self.lstImages[i][3]

    def get_keypoints_location(self, i):
        # To be written to return a dictionary with the proper semantic
        return None

class SFEW(FaceImagesDataset):

    def __init__(self):
        # super(SFEW,self).__init__("SFEW", "/data/lisa/data/faces/SFEW/")
        super(SFEW,self).__init__("SFEW", "faces/SFEW/")
        self.lstImages=[]
        self.keyPointsDict=[]
        emotionsDic = { "Angry":0, "Disgust":1, "Fear":2 , "Happy":3, "Sad":4, "Surprise":5, "Neutral":6}
        sets=["Set1","Set2"]
        for s in sets:
            path = "SFEW-SPI-Release/"+s+"/"
            pathKP = "SFEW-SPI-Annotations/"+s+"/"
            lstemo = os.listdir(self.absolute_base_directory+path) # Emotions
            for em in lstemo:

                # Load the appropriate keypoints file
                dictKP = self.readKeypointsCSV(self.absolute_base_directory+pathKP+em+"/FaceFP_5.txt")

                # Add each image and its keypoints to the lists lstImages and lstKeypoints
                images = os.listdir(self.absolute_base_directory+path+em)
                images.sort()
                for im in images:
                    if im[-4:] == ".jpg":
                        self.lstImages.append([path+em+"/"+im, emotionsDic[em]])
                        self.keyPointsDict.append(dictKP[im])
                    
    def readKeypointsCSV(self,filename):
        keypointsDict = {}
        f = open(filename)
        for line in iter(f):
            content = [dataPair.split(" ") for dataPair in line.split("\t")]
            keypointsDict[content[0][0]] = {"right_eye_pupil": (float(content[1][0]),float(content[1][1])),
                                            "left_eye_pupil": (float(content[2][0]),float(content[2][1])),
                                            "nose_tip": (float(content[3][0]),float(content[3][1])),
                                            "mouth_right_corner": (float(content[4][0]),float(content[4][1])),
                                            "mouth_left_corner": (float(content[5][0]),float(content[5][1]))}
        f.close()
        return keypointsDict

    def __len__(self):
        return len(self.lstImages)

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.lstImages[i][0]

    def get_7emotion_index(self, i):
        index = self.lstImages[i][1]
        if index < 7:
            return index
        return None

    def get_eyes_location(self, i):
        d = self.keyPointsDict[i]
        if d is not None:
            return d['left_eye_pupil'] + d['right_eye_pupil']
        else:
            return None

    def get_keypoints_location(self, i):
        return [self.keyPointsDict[i]]


class MSFDE(FaceImagesDataset):

    def __init__(self):
        super(MSFDE,self).__init__("MSFDE", "faces/MSFDE/MSFDE/MSFDE/")
        self.lstImages=[]
        localEmotions = ["Angry","Happy","Sad","Fear","Disgust","Shame"]
        emotionsDic = { "Angry":0, "Disgust":1, "Fear":2 , "Happy":3, "Sad":4, "Surprise":5, "Neutral":6,"Shame":7}

        FACS= ["4+5+23"," 6+12+25","1+4+15","1+2+5+20+25","9+25","32"]
        lstethnicity = os.listdir(self.absolute_base_directory)

        self.imageIndex = {}
        idx = 0
        for eth in lstethnicity:
            #import pdb;pdb.set_trace()
            if os.path.isdir(self.absolute_base_directory+eth):
                path = eth+"/"
                lstgender= os.listdir(self.absolute_base_directory+path)
                for gen in lstgender:
                    if os.path.isdir(self.absolute_base_directory+eth+"/"+gen):
                        if gen.startswith('Ma'):
                            cgender="Male"
                        else:
                            cgender="Female"
                        path = eth+"/"+gen+"/"
                        images = os.listdir(self.absolute_base_directory+path)
                        for im in images:
                            if im.endswith(".tif"):
                                if im[2] ==' ':
                                    self.lstImages.append([path+im, emotionsDic["Neutral"], None])
                                    self.imageIndex[os.path.basename(im)] = idx
                                    idx += 1
                                else:
                                    try:
                                        self.lstImages.append([path+im, emotionsDic[localEmotions[int(im[2])-1]], FACS[int(im[2])-1],cgender])
                                        self.imageIndex[os.path.basename(im)] = idx
                                        idx += 1
                                    except:
                                        print path+im

    def __len__(self):
        return len(self.lstImages)

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.lstImages[i][0]

    def get_7emotion_index(self, i):
        index = self.lstImages[i][1]
        if index < 7:
            return index
        return None

    def get_gender(self,i):
        return self.lstImages[i][3]

    def get_facs(self, i):
        return self.lstImages[i][2]


class IndianFaceDatabase(FaceImagesDataset):

    def __init__(self):
        super(IndianFaceDatabase,self).__init__("IndianFaceDatabase",
                                                "faces/IndianFaceDatabase/dbase/")

        # Get the path of all the image files in subfolders of the baseFolder
        self.images = []
        self.subjectIdxByImage = []
        self.subjectStrings = []
        self.sexes = []
        self.imageIndex = {}

        filenames = glob.glob(self.absolute_base_directory + "*/*/*.jpg")
        filenames.sort()
        filenames = [f.replace(self.absolute_base_directory,'') for f in filenames]
        
        for idx, filename in enumerate(filenames):
            
            relPathToImage = filename
            self.images.append(relPathToImage)

            # Remove the name of the image to keep only the sex and id
            # of the subject
            subjectString = '/'.join(filename.split('/')[:-1])
                    
            if subjectString not in self.subjectStrings:
                self.subjectStrings.append(subjectString)

            subjectId = self.subjectStrings.index(subjectString)
            self.subjectIdxByImage.append(subjectId)
            
            sex = filename.split('/')[0]
            if sex == "males":
                self.sexes.append("Male")
            else:
                self.sexes.append("Female")

            imgName = os.path.basename(filename)
            self.imageIndex[os.path.join(subjectString, imgName)] = idx

        # WARNING : Extract emotion, and pose information from 
        # the images. This part is an approximation because those information,
        # while present in the dataset, are not rigorously described for the
        # dataset. If you want them you have to infer them from the filenames
        # and the order of the images. The problem is that the filenames are
        # not consistent throughout the dataset, specially for male subjects.        
        self.emotions = [None,] * len(self.images)
        self.extendedEmotions = [None,] * len(self.images)
        self.poses = [None,] * len(self.images)
        
        for k in range(self.get_n_subjects()):
            imgIdxs = self.get_face_examples_for_subject(k)
            imgIdxs.sort()
            
            if len(imgIdxs) == 11:
                # There is no way we can infer the emotion or
                # pose reliably from the images if there are not
                # 11 of them exactly.
            
                if "001" in self.images[imgIdxs[0]]:
                    self.poses[imgIdxs[0]] = 4
                    
                if "002" in self.images[imgIdxs[1]]:
                    self.poses[imgIdxs[1]] = 3
                    
                if "003" in self.images[imgIdxs[2]]:
                    self.poses[imgIdxs[2]] = 5
                    
                if "004" in self.images[imgIdxs[3]]:
                    self.poses[imgIdxs[3]] = 1
                    
                if "005" in self.images[imgIdxs[4]]:
                    self.poses[imgIdxs[4]] = 0
                    
                if "006" in self.images[imgIdxs[5]]:
                    self.poses[imgIdxs[5]] = 2
                    
                if "007" in self.images[imgIdxs[6]]:
                    self.poses[imgIdxs[6]] = 7
                    
                if "008" in self.images[imgIdxs[7]]:
                    self.emotions[imgIdxs[7]] = 6
                    
                if "009" in self.images[imgIdxs[8]]:
                    self.emotions[imgIdxs[8]] = 3 
                    
                if "010" in self.images[imgIdxs[9]]:
                    self.extendedEmotions[imgIdxs[9]] = "laughter" 
                    
    def __len__(self):
        return len(self.images)

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]

    def get_original_image_path_relative_to_base_directory(self, i):
        if i >= 0 and i < len(self.images):
            return self.images[i]
        else:
            return None

    def get_n_subjects(self):
        return len(self.subjectStrings)

    def get_id_of_kth_subject(self, k):
        if k >= 0 and k < len(self.subjectStrings):
            return self.subjectStrings[k]
        else:
            return None

    def get_subject_id_of_ith_face(self, i):
        if i >= 0 and i < len(self.images):
            return self.subjectStrings[self.subjectIdxByImage[i]]
        else:
            return None
            
    def get_detailed_emotion_label(self, i):
        emotion = self.get_7emotion_label(i)
        
        if emotion != None:
            return emotion
        else:
            return self.extendedEmotions[i]

    def get_7emotion_index(self, i):
        if i >= 0 and i < len(self.images):
            return self.emotions[i]
        else:
            return None
        
    def get_head_pose(self, i):
        if i >= 0 and i < len(self.images):
            return self.poses[i]
        else:
            return None
        
    def get_gender(self,i):
        if i >= 0 and i < len(self.images):
            return self.sexes[i]
        else:
            return None


class Jaffe(FaceImagesDataset):

    # Dictionnary of the abbreviations used for each emotion in the dataset
    emotionAbbreviations = {'AN' : 0, 'DI' : 1, 'FE' : 2, 'HA' : 3,
                            'SA' : 4, 'SU' : 5, 'NE' : 6}

    def __init__(self):
        super(Jaffe,self).__init__("JAFFE", "faces/jaffe/JAFFE/")

        # Get the path of all the image files in subfolders of the baseFolder
        self.images = []
        self.subjectIdxByImage = []
        self.subjectStrings = []
        self.emotionIdxByImage = []
        self.imageIndex = {}

        filenames = os.listdir(self.absolute_base_directory)
        filenames.sort() # Sort to ensure consistent order across platforms

        idx = 0
        for filename in filenames:
            if filename.endswith(".tiff"):
                subjectString = filename.split(".")[0]
                emotionAbbrev = filename.split(".")[1][:2]

                imgName = os.path.basename(filename)
                self.imageIndex[imgName] = idx

                if subjectString not in self.subjectStrings:
                    self.subjectStrings.append(subjectString)

                self.images.append(filename)

                emotionIdx = self.emotionAbbreviations[emotionAbbrev]
                self.emotionIdxByImage.append(emotionIdx)

                subjectId = self.subjectStrings.index(subjectString)
                self.subjectIdxByImage.append(subjectId)

                idx += 1

    def __len__(self):
        return len(self.images)

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]

    def get_original_image_path_relative_to_base_directory(self, i):
        if i >= 0 and i < len(self.images):
            return self.images[i]
        else:
            return None

    def get_n_subjects(self):
        return len(self.subjectStrings)

    def get_id_of_kth_subject(self, k):
        if k >= 0 and k < len(self.subjectStrings):
            return self.subjectStrings[k]
        else:
            return None

    def get_subject_id_of_ith_face(self, i):
        if i >= 0 and i < len(self.images):
            return self.subjectStrings[self.subjectIdxByImage[i]]
        else:
            return None

    def get_7emotion_index(self, i):
        if i >= 0 and i < len(self.images):
            return self.emotionIdxByImage[i]
        else:
            return None
            
    def get_gender(self,i):
        if i >= 0 and i < len(self.images):
            # Subjects are all female
            return "Female"
        else:
            return None


class Jacfee(FaceImagesDataset):

    jacfee_emotion_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

    def __init__(self):
        super(Jacfee,self).__init__("JACFEE", "faces/JACFEE/standard_expressor_set/")

        # Load the csv file listing.txt which describes the whoke content of
        # the dataset
        f = open(self.absolute_base_directory + "listing.txt")
        lines = f.readlines()
        f.close()

        # Process each line of listing.txt which describes an image of the
        # dataset
        self.images = []
        self.subjectIdxByImage = []
        self.subjectStrings = []
        self.emotionIdxByImage = []
        self.facsByImage = []
        self.sexes = []

        for line in lines[1:]:

            attributes = line.split(",")

            subjectString = attributes[0]
            imageName = attributes[1]
            race = attributes[2]
            sex = attributes[3]
            emotion = attributes[4]
            facs = attributes[5]

            if subjectString not in self.subjectStrings:
                    self.subjectStrings.append(subjectString)

            self.images.append(subjectString + "/" + imageName + ".jpg")

            subjectId = self.subjectStrings.index(subjectString)
            self.subjectIdxByImage.append(subjectId)

            if sex == "male":
                self.sexes.append("Male")
            else:
                self.sexes.append("Female")          
            
            if emotion in self.jacfee_emotion_names:
                self.emotionIdxByImage.append(
                                    self.jacfee_emotion_names.index(emotion))
            else:
                self.emotionIdxByImage.append(None)

            # Assemble the facs dictionnary for the image
            imageFACS = None
            if facs != "*":
                imageFACS = {}
                for fac in facs.strip().split("+"):
                    fac = fac.strip()

                    if "(" in fac:
                        fac = fac[:fac.index("(")]
                    assert len(fac) == 2 or len(fac) == 3

                    imageFACS[fac[:-1]] = fac[-1]

            self.facsByImage.append(imageFACS)

    def __len__(self):
        return len(self.images)

    def get_original_image_path_relative_to_base_directory(self, i):
        if i >= 0 and i < len(self.images):
            return self.images[i]
        else:
            return None

    def get_n_subjects(self):
        return len(self.subjectStrings)

    def get_id_of_kth_subject(self, k):
        if k >= 0 and k < len(self.subjectStrings):
            return self.subjectStrings[k]
        else:
            return None

    def get_subject_id_of_ith_face(self, i):

        if i >= 0 and i < len(self.images):
            return self.subjectStrings[self.subjectIdxByImage[i]]
        else:
            return None

    def get_7emotion_index(self, i):
        if i >= 0 and i < len(self.images):
            return self.emotionIdxByImage[i]
        else:
            return None

    def get_facs(self, i):
        if i >= 0 and i < len(self.images):
            return self.facsByImage[i]
        else:
            return None
            
    def get_gender(self,i):
        if i >= 0 and i < len(self.images):
            return self.sexes[i]
        else:
            return None


class ArFace(FaceImagesDataset):

    ArFace_labels = ["neutral", "smile", "anger", "scream", "left_light_on", "right_light_on", "all_side_lights_on",
                     "wearing_sun_glasses", "wearing_sun_glasses_and_left_light_on",
                     "wearing_sun_glasses_and_right_light_on", "wearing_scarf", "wearing_scarf_left_light_on",
                     "wearing_scarf_right_light_on"]

    def __init__(self):
        super(ArFace, self).__init__("ArFaceDatabase", "faces/AR_Face_Database")

        self.imageRelativePath = []  # Relative path to images
        self.subjectId = []  # The "name" of the subject
        self.subjectIdxByImage = []  # The index in self.subjectId that returns the subject in the image
        self.subjectGender = []  # The gender of the subject
        self.subjectLabelIndex = []
        self.keyPointsDict = []

        imageFiles = glob.glob(os.path.join(self.absolute_base_directory, "dbf*/*.png"))
        imageFiles.sort()  # Make sure we always return the same order

        for fullPath in imageFiles:
            # Image relative path
            relPath = os.path.relpath(fullPath, self.absolute_base_directory)
            imgName = os.path.basename(relPath)
            self.imageRelativePath.append(relPath)

            # Subject gender
            subjectGender = "m" if imgName[0] == "m" else "f"
            self.subjectGender.append(subjectGender)

            # SubjectId
            subjectId = imgName.split("-")[0] + "-" + imgName.split("-")[1]
            if subjectId not in self.subjectId:
                self.subjectId.append(subjectId)

            # SubjectIdxByImage
            self.subjectIdxByImage.append(self.subjectId.index(subjectId))

            # Label index
            labelIdx = int(imgName.split("-")[2].replace(".png", ""))
            labelIdx /= 1 if labelIdx <= 13 else 2
            self.subjectLabelIndex.append(labelIdx)

            # Keypoints dictionary
            keyPointsFileName = imgName.replace(".png", ".pts")
            if keyPointsFileName[-6] == "-":  # 02 instead of 2 in key points file name
                keyPointsFileName = keyPointsFileName[:-5] + "0" + keyPointsFileName[-5:]
            keyPointsPath = os.path.join(self.absolute_base_directory, "points_22", subjectId,
                                         keyPointsFileName)
            d = self.readKeyPointsFromArFaceFile(keyPointsPath)
            if dict is not None:
                self.keyPointsDict.append(d)
            else:
                self.keyPointsDict.append(None)

    def readKeyPointsFromArFaceFile(self, keyPointsFileName):
        """
        Reads the ArFace keypoints file and returns the KeyPoint dictionary
        """
        if not os.path.exists(keyPointsFileName):
            return {}

        with open(keyPointsFileName, 'r') as keyPointsFile:
            keyPointsLines = keyPointsFile.readlines()

        keyPoints = {"left_eye_pupil": (float(keyPointsLines[4].split(" ")[0]), float(keyPointsLines[4].split(" ")[1])),
                     "right_eye_pupil": (float(keyPointsLines[3].split(" ")[0]), float(keyPointsLines[3].split(" ")[1])),
                     "left_eye_inner_corner": (float(keyPointsLines[14].split(" ")[0]), float(keyPointsLines[14].split(" ")[1])),
                     "left_eye_outer_corner": (float(keyPointsLines[15].split(" ")[0]), float(keyPointsLines[15].split(" ")[1])),
                     "right_eye_inner_corner": (float(keyPointsLines[13].split(" ")[0]), float(keyPointsLines[13].split(" ")[1])),
                     "right_eye_outer_corner": (float(keyPointsLines[12].split(" ")[0]), float(keyPointsLines[12].split(" ")[1])),
                     "left_eyebrow_inner_end": (float(keyPointsLines[9].split(" ")[0]), float(keyPointsLines[9].split(" ")[1])),
                     "left_eyebrow_outer_end": (float(keyPointsLines[10].split(" ")[0]), float(keyPointsLines[10].split(" ")[1])),
                     "right_eyebrow_inner_end": (float(keyPointsLines[8].split(" ")[0]), float(keyPointsLines[8].split(" ")[1])),
                     "right_eyebrow_outer_end": (float(keyPointsLines[7].split(" ")[0]), float(keyPointsLines[7].split(" ")[1])),
                     "nose_tip": (float(keyPointsLines[17].split(" ")[0]), float(keyPointsLines[17].split(" ")[1])),
                     "mouth_left_corner": (float(keyPointsLines[6].split(" ")[0]), float(keyPointsLines[6].split(" ")[1])),
                     "mouth_right_corner": (float(keyPointsLines[5].split(" ")[0]), float(keyPointsLines[5].split(" ")[1])),
                     "mouth_top_lip": (float(keyPointsLines[20].split(" ")[0]), float(keyPointsLines[20].split(" ")[1])),
                     "mouth_bottom_lip": (float(keyPointsLines[21].split(" ")[0]), float(keyPointsLines[21].split(" ")[1]))}

        return keyPoints

    def __len__(self):
        return len(self.imageRelativePath)

    def get_original_image_path_relative_to_base_directory(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.imageRelativePath[i]
        else:
            return None

    def get_n_subjects(self):
        return len(self.subjectId)

    def get_id_of_kth_subject(self, k):
        if 0 <= k < len(self.subjectId):
            return self.subjectId[k]
        else:
            return None

    def get_subject_id_of_ith_face(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.subjectId[self.subjectIdxByImage[i]]
        else:
            return None

    def get_7emotion_index(self, i):
        """
        Returns an emotion index if the ArFace emotions matches the standard ones
        """
        if 0 <= i < len(self.imageRelativePath):
            labelIdx = self.subjectLabelIndex[i]
            if labelIdx == 0:
                return 6  # Neutral
            if labelIdx == 2:
                return 0  # Anger
            if labelIdx == 1:
                return 3  # Happy
            return None
        else:
            return None

    def get_gender(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.subjectGender[i]
        else:
            return None

    def get_light_source_direction(self, i):
        if 0 <= i < len(self.imageRelativePath):
            labelIdx = self.subjectLabelIndex[i]
            label = ArFace.ArFace_labels[labelIdx]
            if "left" in label:
                return "left"
            elif "right" in label:
                return "right"
            else:
                return None
        else:
            return None

    def get_eyes_location(self, i):
        d = self.keyPointsDict[i]
        if len(d) != 0:
            return [d['right_eye_pupil'][0], d['right_eye_pupil'][1], d['left_eye_pupil'][0], d['left_eye_pupil'][1]]
        else:
            return None

    def get_keypoints_location(self, i):
        return [self.keyPointsDict[i]]


class Kdef(FaceImagesDataset):

    def __init__(self):
        super(Kdef, self).__init__("Kdef", "faces/KDEF_and_AKDEF/KDEF/")

        self.imageRelativePath = []  # Relative path to images
        self.subjectId = []  # The "name" of the subject
        self.subjectIdxByImage = []  # The index in self.subjectId that returns the subject in the image
        self.subjectGender = []  # The gender of the subject
        self.imageNameNoExtension = []  # The image name because it contains many information
        self.imageIndex = {}

        imageFiles = glob.glob(os.path.join(self.absolute_base_directory, "*/*.JPG"))
        imageFiles.sort()  # Make sure we always return the same order

        for idx, fullPath in enumerate(imageFiles):
            # Image relative path
            relPath = os.path.relpath(fullPath, self.absolute_base_directory)
            imgName = os.path.basename(relPath)
            self.imageRelativePath.append(relPath)
            self.imageNameNoExtension.append(imgName.replace(".JPG", ""))
            self.imageIndex[imgName] = idx

            # Subject gender
            subjectGender = "M" if imgName[1] == "M" else "F"
            self.subjectGender.append(subjectGender)

            # SubjectId
            subjectId = imgName[1:4]
            if subjectId not in self.subjectId:
                self.subjectId.append(subjectId)

            # SubjectIdxByImage
            self.subjectIdxByImage.append(self.subjectId.index(subjectId))

    def __len__(self):
        return len(self.imageRelativePath)

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]

    def get_original_image_path_relative_to_base_directory(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.imageRelativePath[i]
        else:
            return None

    def get_n_subjects(self):
        return len(self.subjectId)

    def get_id_of_kth_subject(self, k):
        if 0 <= k < len(self.subjectId):
            return self.subjectId[k]
        else:
            return None

    def get_subject_id_of_ith_face(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.subjectId[self.subjectIdxByImage[i]]
        else:
            return None

    def get_7emotion_index(self, i):
        if 0 <= i < len(self.imageRelativePath):
            imageName = self.imageNameNoExtension[i]
            emotionCode = imageName[4:6]
            #["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            if emotionCode == "AF":
                return 2  # fear
            if emotionCode == 'AN':
                return 0  # Anger
            if emotionCode == 'DI':
                return 1  # Disgust
            if emotionCode == "HA":
                return 3  # Happy
            if emotionCode == 'NE':
                return 6  # Neutral
            if emotionCode == 'SA':
                return 4  # Sad
            if emotionCode == 'SU':
                return 5  # Surprise
            return None
        else:
            return None

    def get_gender(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.subjectGender[i]
        else:
            return None

    def get_head_pose(self, i):
        if 0 <= i < len(self.imageRelativePath):
            imageName = self.imageNameNoExtension[i]
            lightCode = imageName[6:8]
            #   0 1 2
            # 9 3 4 5 10
            #   6 7 8
            if lightCode == "FL":
                return 9
            if lightCode == "HL":
                return 6
            if lightCode == "S":
                return 4
            if lightCode == "HR":
                return 5
            if lightCode == "FR":
                return 10
            else:
                return None
        else:
            return None


class NimStim(FaceImagesDataset):

    def __init__(self):
        super(NimStim, self).__init__("NimStim", "faces/NimStim")

        self.imageRelativePath = []  # Relative path to images
        self.subjectId = []  # The "name" of the subject
        self.subjectIdxByImage = []  # The index in self.subjectId that returns the subject in the image
        self.subjectGender = []  # The gender of the subject
        self.imageNameNoExtension = []  # The image ** in lowercase ** name because it contains many information
        self.imageIndex = {}
        imageFiles = glob.glob(os.path.join(self.absolute_base_directory,
                                            "NimStim.NOBACKUP/Crop-White Background/*.[bB][mM][pP]"))
        imageFiles.sort()  # Make sure we always return the same order

        for idx, fullPath in enumerate(imageFiles):
            # Image relative path
            relPath = os.path.relpath(fullPath, self.absolute_base_directory)
            imgName = os.path.basename(relPath).lower()
            self.imageIndex[imgName.lower()] = idx  # We put the lower case because some images have .bmp others .BMP
            self.imageRelativePath.append(relPath)
            self.imageNameNoExtension.append(imgName.replace(".bmp", ""))

            # Subject gender
            subjectGender = "M" if imgName[2] == "m" else "F"
            self.subjectGender.append(subjectGender)

            # SubjectId
            subjectId = imgName[0:3]
            if subjectId not in self.subjectId:
                self.subjectId.append(subjectId)

            # SubjectIdxByImage
            self.subjectIdxByImage.append(self.subjectId.index(subjectId))

    def __len__(self):
        return len(self.imageRelativePath)

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName.lower()]

    def get_original_image_path_relative_to_base_directory(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.imageRelativePath[i]
        else:
            return None

    def get_n_subjects(self):
        return len(self.subjectId)

    def get_id_of_kth_subject(self, k):
        if 0 <= k < len(self.subjectId):
            return self.subjectId[k]
        else:
            return None

    def get_subject_id_of_ith_face(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.subjectId[self.subjectIdxByImage[i]]
        else:
            return None

    def get_7emotion_index(self, i):
        if 0 <= i < len(self.imageRelativePath):
            imageName = self.imageNameNoExtension[i]
            emotionCode = imageName[4:6]
            #["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            if emotionCode == "af":
                return 2  # fear
            if emotionCode == 'an':
                return 0  # Anger
            if emotionCode == 'di':
                return 1  # Disgust
            if emotionCode == "ha":
                return 3  # Happy
            if emotionCode == 'ne':
                return 6  # Neutral
            if emotionCode == 'sa':
                return 4  # Sad
            if emotionCode == 'su':
                return 5  # Surprise
            return None
        else:
            return None

    def get_gender(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.subjectGender[i]
        else:
            return None

    def get_head_pose(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return 4
        else:
            return None

    def get_is_mouth_opened(self, i):
        if 0 <= i < len(self.imageRelativePath):
            imageName = self.imageNameNoExtension[i]
            openClose = imageName[7]
            if openClose == "c":
                return False
            if openClose in {'o', 'x'}:  # We consider exuberant as open
                return True
            return None
        else:
            return None


class Pofa(FaceImagesDataset):

    def __init__(self):
        super(Pofa, self).__init__("Pofa", "faces/POFA/")

        self.imageRelativePath = []  # Relative path to images
        self.imageEmotions = []  # The emotion in each images
        self.imageIndex = {}

        imageFiles = glob.glob(os.path.join(self.absolute_base_directory, "POFA/*.TIF"))
        imageFiles.sort()  # Make sure we always return the same order

        emotionDict = self.loadEmotionDict()

        for idx, fullPath in enumerate(imageFiles):
            # Image relative path
            relPath = os.path.relpath(fullPath, self.absolute_base_directory)
            imgName = os.path.basename(relPath)
            self.imageIndex[imgName] = idx
            self.imageRelativePath.append(relPath)

            # Emotion of the image
            if emotionDict is None:
                self.imageEmotions.append(None)
            else:
                number = int(imgName.replace('.TIF', ''))
                self.imageEmotions.append(emotionDict[number])

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]


    def loadEmotionDict(self):
        """
        Load the emotion dictionary {imageNumber : emotion}
        """
        fileName = os.path.join(self.absolute_base_directory, "POFA/simple_labels.txt")
        d = {}
        with open(fileName) as f:
            for line in f:
                (emotion, imageNumber) = line.split(':')
                emotion = emotion.strip().lower()

                if emotion == "anger":
                    emotion = 0
                elif emotion == "disgust":
                    emotion = 1
                elif emotion == "fear":
                    emotion = 2
                elif emotion == "happy":
                    emotion = 3
                elif emotion == "sad":
                    emotion = 4
                elif emotion == "surprise":
                    emotion = 5
                elif emotion == "neutral":
                    emotion = 6
                else:
                    return None  # The file changed since we last coded this

                imageNumber = imageNumber.split(',')
                for i in imageNumber:
                    d[int(i)] = emotion

        return d

    def __len__(self):
        return len(self.imageRelativePath)

    def get_original_image_path_relative_to_base_directory(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.imageRelativePath[i]
        else:
            return None

    def get_7emotion_index(self, i):
        if 0 <= i < len(self.imageRelativePath):
            return self.imageEmotions[i]
        else:
            return None


#### List of dataset names, constructor, description

datasets_constructors_list = [
    ("TorontoFaceDataset", TorontoFaceDataset, "48x48 Toronto Face Dataset images. A mix of several other face expression datasets."),
    ("StaticCKPlus", StaticCKPlus, "First (neutral) abnd last (most emotive) image from all CKPlus sequences. (part of TFD)"),
    ("SFEW", SFEW, " (part of TFD)"),
    ("MSFDE", MSFDE, " (part of TFD)"),
    ("IndianFaceDatabase", IndianFaceDatabase, " (part of TFD)"),
    ("Jaffe", Jaffe, " (part of TFD)"),
    ("Jacfee", Jacfee, " (part of TFD)"),
    ("ArFace", ArFace, " (part of TFD)"),
    ("Kdef", Kdef, " (part of TFD)"),
    ("NimStim", NimStim, " (part of TFD)"),
    ("Pofa", Pofa, " (part of TFD)") ]


