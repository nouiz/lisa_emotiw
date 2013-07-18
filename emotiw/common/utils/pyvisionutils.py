
import cv
import pyvision as pv
import pyvision.face.CascadeDetector as cd
from emotiw.common.datasets.faces.faceimages import bbox_size

face_detect = None

def pyvision_detect_faces_bboxes(image):
    """Will return a (possibly empty) list of bounding boxes as tuples (x1,y1,x2,y2)
    containing coordinates of upper left corner x1,y1 and lower right corner x2,y2.
    List will be sorted in order of decreasingbounding box sizes."""

    bboxes = pyvision_detect_faces(image)
    if bboxes is None:
        return []
    bboxes = [ (b.x, b.y, b.x+b.w, b.y+b.h) for b in bboxes ]
    bboxes.sort(key=bbox_size, reverse=True)
    return bboxes

def cv2ndarray_to_iplimage(source):
    """source is numpy array returned by cv2"""
    if len(source.shape)!=3 or source.shape[2]!=3:
        raise ValueError("cv2ndarray_to_iplimage currently only supports 3 dimensional ndarrays with depth 3")
    h,w,d = source.shape
    bitmap = cv.CreateImageHeader((w, h), cv.IPL_DEPTH_8U, 3)
    cv.SetData(bitmap, source.tostring(), 
           source.dtype.itemsize * 3 * source.shape[1])
    return bitmap

def pyvision_detect_faces(image):
    global face_detect
    if face_detect is None:
        face_detect = cd.CascadeDetector()

    # print "DETECTING FACES"
    if not hasattr(image,"asOpenCV"):
        if hasattr(image, "shape"): # numpy ndarray, assuming it comes from cv2
            image = cvu.cv2ndarray_to_iplimage(image)
        image = pv.Image(image,bw_annotate=True)
    faces = face_detect(image)
    # print "DETECTING FACES DONE."
    return faces

