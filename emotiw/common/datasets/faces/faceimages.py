import os
import cv

from emotiw.common.utils.pathutils import locate_data_path

#sys.path.append(os.getcwd()+"/../../../vincentp")
#from preprocess_face import * # getEyesPositions,getFaceBoundingBox

basic_7emotion_names = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class FaceImagesDataset(object):
    """
    Base class defining a standard interface to access datasets containing static images of faces and all associated info.
    """

    def __init__(self, dataset_name, relative_base_directory=None):
        """
        Initilizes a dataset with the given name, and whose
        constituent files are located in the specified
        directory. This directory is specified only relative to
        some standard dataset root location (found with locate_data_path ) 
        """
        self.dataset_name = dataset_name
        if relative_base_directory is not None:
            self.relative_base_directory = relative_base_directory
            self.absolute_base_directory = locate_data_path(relative_base_directory)

    def get_name(self):
        return self.dataset_name

    def __len__(self):
        """
        Returns the total number of face examples in the dataset.
        """
        raise NotImplementedError(str(type(self))+" does not implement __len__.")

    def __getitem__(self, i):
        """
        Returns an object that exposes all properties of the ith face example
        Technically it will return an instance of FaceDatasetExample on which
        accessing properties will be handled by calling the corresponding get_... method
        of the current dataset, with parameter i.
        """
        
        if isinstance(i,int):
            if i<0 or i>=self.__len__():
                raise IndexError()
            return FaceDatasetExample(self, i)
                
        elif isinstance(i,slice):
            return FaceImagesSubset(self,range(i.start, i.stop, i.step))

        else: # assume list of indexes
            li = [idx for idx in i] # build list fro any iterable
            return FaceImagesSubset(self,li)

    def get_original_image(self, i):
        filepath = self.get_original_image_path(i)
        img = cv.LoadImage(filepath)
        return img

    def get_original_image_path(self,i):
        """
        Returns a full path from where to load the original image
        """
        relpath = self.get_original_image_path_relative_to_base_directory(i)
        return os.path.join(self.absolute_base_directory, relpath)

    def get_original_image_path_relative_to_base_directory(self, i):
        """
        Returns the relative path that needs to be concatenated to the
        relative_base_directory (which locates this dataset's directory) 
        in order to access the image file containing the ith face example.
        """
        raise NotImplementedError()

    def get_index_from_image_filename(self, imgFileName):
        """
        Returns the index number of the image named 'imgFileName' so that we can call methods that take
        "ith face" index as a parameter.

        imgFileName is usually only the filename. When dataset use the same file names in different folders for various
        images, then imgFileName is the shortest path to distinguish all images.
        """

    def get_bbox(self, i):
        """
        Returns a list of a bounding box around the faces in the ith image,
        as recorded (or precomputed) in the dataset. Returns None if not available.

        This is necessary for datasets with images that contain several faces,
        And relevant for databases that could be used for training a face detector.

        Default version returns the first one of get_orginal_bbox, get_picasa_bbox, get_opencv_bbox that is not None
        Subclasses should override one or several of these as appropriate.
        """
        bbox = self.get_original_bbox(i)
        if bbox is None:
            bbox = self.get_picasa_bbox(i)
            if bbox is None:
                bbox = self.get_opencv_bbox(i)
        return bbox

    def get_picasa_bbox(self, i):
        return None
    
    def get_opencv_bbox(self, i):
        return None

    def get_original_bbox(self, i):
        return None
    
    def get_eyes_location(self, i):
        """
        Returns the location of the center of the two eyes, in imge coordinates, as a tuple
        (left_x, left_y, right_x, right_y) in that order or None if not available.
        left means the leftmost eye on the image (usually corresponding to the person's right eye).
        Coordinate system has its origin in the upper left corner of the image (horizontal_offset_in_pixels,vertical_offset_in_pixels).
        """
        return None

    def get_keypoints_location(self, i):
        """
        Returns a dictionary of keypoint_names -> (x,y) image coordinates (or None if not available).
        Coordinate system has its origin in the upper left corner of the image
            (horizontal_offset_in_pixels, vertical_offset_in_pixels).

        Possible names for the keypoints should be one of FGNet definition :
        *** Left = left of subject therefore at the right of the image in frontal view ***
        left_eye_pupil
        right_eye_pupil
        left_eye_center
        right_eye_center
        left_eye_inner_corner
        left_eye_outer_corner
        right_eye_inner_corner
        right_eye_outer_corner
        left_eyebrow_inner_end
        left_eyebrow_outer_end
        right_eyebrow_inner_end
        right_eyebrow_outer_end
        nose_tip
        mouth_left_corner
        mouth_right_corner
        mouth_center_top_lip
        mouth_center_bottom_lip
        """
        return None

    def get_n_subjects(self):
        """
        Returns how many different subjects are in the database (or None if unknown)
        """
        return None

    def get_id_of_kth_subject(self, k):
        """
        Retruns a string uniquely identifying the kth subject from this database (or None if unknown)
        """
        return None

    def get_face_examples_for_subject(self, k):
        """
        Returns a list of indexes of all face examples associated with the kth subject.
        Returns None if id info not available.
        """
        id = self.get_id_of_kth_subject(k)
        if id is None:
            return None
        matching_example_indexes =  [ i for i in range(len(self)) if self.get_subject_id_of_ith_face(i)==id ] 
        return matching_example_indexes

    def get_subject_id_of_ith_face(self, i):
        """
        Returns a string that uniquely identifies the identity of the subject in the ith face of this dataset.
        Returns None if this information is not available.
        """
        return None

    def get_detailed_emotion_label(self, i):
        """
        Returns a string containing detailed emotion label associateed to the ith face
        returns None is not available.

        Default version calls get_7emotion_label.
        """
        return self.get_7emotion_label(i)

    def get_7emotion_label(self, i):
        """
        Returns a string containing one of the 7 basic emotion labels associatedto the ith face examples
        returns None is not available.
        'anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'

        Default version returns the name associated to numerical index returned by get_7emotion_index
        """
        emo_index = self.get_7emotion_index(i)
        if emo_index is None or emo_index<0:
            return None
        else:
            return basic_7emotion_names[emo_index]

    def get_7emotion_index(self, i):
        """
        Returns the emotion index for the ith image.
        Indexes are : 
            0 : anger
            1 : disgust
            2 : fear
            3 : happy
            4 : sad
            5 : surprise
            6 : neutral
        Returns None if this information is not available or the provided emotion is unrelated
        to the 7 emotions defined above.
        """
        return None

    def get_facs(self, i):
        """
        Returns the facial action coding system for the ith image (or None if not available).
        http://en.wikipedia.org/wiki/Facial_Action_Coding_System

        Returns a dictionary:
        keys are the action unit codes for the action units that were observed,
        its associated values are intensity scoring 'A', 'B', 'C', 'D', or 'E' or None if the intensity score is not known.
        """
        return None

    def get_head_pose(self, i):
        """
        Returns the head pose relative to the camera (or None if not available)
        
        The pose is described, for now, as an integer value between 0 and 8
        indicating the direction the subject's head is facing. Each of these
        integer values is associated with a quadrant as shown in the square
        below :
          0 1 2
        9 3 4 5 10
          6 7 8
        
        For instance : 
        1 means the subject is looking up
        3 means the subject is looking to his right (left of the camera) at about 45 degrees
        4 means the subject is looking directly at the camera 
        8 means the subject is looking down to his left (right of the camera)
        9 means profile view (the subject is looking to his right (left of the camera) at about 90 degrees
        ...
        """
        return None

    def get_light_source_direction(self, i):
        """
        Returns the direction of the light source (or None if not available)
        """
        return None

    def get_gaze_direction(self, i):
        """
        Returns the direction in which the person was looking (or None if not available)
        """
        return None

    def get_gender(self,i):
        """
        Returns either male of female if available
        """
        return None

    def get_is_mouth_opened(self, i):
        """
        Returns True if the mouth is opened. False if it's not.
        Returns None if this is not defined on the dataset
        """
        return None

    def get_standard_train_test_splits(self):
        """Returns a list of pairs (train_indexes, test_indexes) where train_indexes and test_indexes are themselves lists or integer ndarrays
        (containing indexes that can be passed to e.g. get_original_image_path)
        Returns None if not available"""
        return None

# Helper class


class FaceImagesSubset(FaceImagesDataset):
    
    def __init__(self, img_dataset, indices, dataset_name=None):
        if dataset_name is None:
            dataset_name = "subset of "+img_dataset.get_name()
        super(FaceImagesSubset,self).__init__(dataset_name)
        self.img_dataset = img_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
        
    def get_original_image_path(self,i):
        return self.img_dataset.get_original_image_path(self.indices[i])
    
    def get_original_image_path_relative_to_base_directory(self, i):
        return self.img_dataset.get_original_image_path_relative_to_base_directory(self.indices[i])
    
    def get_bbox(self, i):
        return self.img_dataset.get_bbox(self.indices[i])
    
    def get_eyes_location(self, i):
        return self.img_dataset.get_eyes_location(self.indices[i])
    
    def get_keypoints_location(self, i):
        return self.img_dataset.get_keypoints_location(self.indices[i])    
    
    def get_subject_id_of_ith_face(self, i):
        return self.img_dataset.get_subject_id_of_ith_face(self.indices[i])
    
    def get_detailed_emotion_label(self, i):
        return self.img_dataset.get_detailed_emotion_label(self.indices[i])
    
    def get_7emotion_label(self, i):
        return self.img_dataset.get_7emotion_label(self.indices[i])
    
    def get_7emotion_index(self, i):
        return self.img_dataset.get_7emotion_index(self.indices[i])
    
    def get_facs(self, i):
        return self.img_dataset.get_facs(self.indices[i])
    
    def get_head_pose(self, i):
        return self.img_dataset.get_head_pose(self.indices[i])
    
    def get_light_source_direction(self, i):
        return self.img_dataset.get_light_source_direction(self.indices[i])
    
    def get_gaze_direction(self, i):
        return self.img_dataset.get_gaze_direction(self.indices[i])
    
    def get_gender(self,i):
        return self.img_dataset.get_gender(self.indices[i])
    
    def get_is_mouth_opened(self, i):
        return self.img_dataset.get_is_mouth_opened(self.indices[i])

    def _get_subject_ids(self):
        if hasattr(self,subject_ids):
            return self.subject_ids

        subject_ids = []
        for i in xrange(len(self)):
            id = self.get_subject_id_of_ith_face(i)
            if id is not None and id not in subject_ids:
                subject_ids.append(id)
        if len(subject_ids)==0:
            self.subject_ids = None
        else:
            self.subject_ids = subject_ids
        return self.subject_ids
    
    def get_n_subjects(self):
        """
        Returns how many different subjects are in the database (or None if unknown)
        """
        subject_ids = self._get_subject_ids()
        if subject_ids is None:
            return None
        return len(subject_ids)

    def get_id_of_kth_subject(self, k):
        """
        Retruns a string uniquely identifying the kth subject from this database (or None if unknown)
        """
        subject_ids = self._get_subject_ids()
        if subject_ids is None:
            return None
        return subject_ids[k]


class FaceDatasetExample(object):
    
    property_names = frozenset([
        "original_image",
        "original_image_path",
        "original_image_path_relative_to_base_directory",
        "bbox",
        "eyes_location",
        "keypoints_location",
        "subject_id_of_ith_face",
        "detailed_emotion_label",
        "7emotion_label",
        "7emotion_index",
        "facs",
        "head_pose",
        "light_source_direction",
        "gaze_direction",
        "gender",
        "is_mouth_opened"
        ])
    
    def __init__(self, dataset, i):
        self._dataset = dataset
        self._i = i
        
    def __getattr__(self, property_name):
        if property_name in FaceDatasetExample.property_names:            
            dataset_method = getattr(self._dataset, "get_"+property_name)
            return dataset_method(i)
        else:
            raise AttributeError()

