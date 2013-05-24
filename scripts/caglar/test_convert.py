import os
from convert_to_h5file import *

bbox_path = "/data/lisa/data/faces/GoogleDataset/images/facesCoordinates/"
faces_path = "/data/lisa/data/faces/GoogleDataset/images/"

def test1():
    dirs = os.walk(faces_path).next()[1]

    images = get_images(faces_path, dirs)
    print images["1"]

def test2():
    bbox_dirs = os.walk(bbox_path).next()[1]

    bboxes = get_bounding_boxes(bbox_path, bbox_dirs)
    print bboxes["1"]

def test3():
    save_img_data(faces_path, bbox_path)

def test4():
    img_path = "/data/lisa/data/faces/GoogleDataset/images/"
    bbox_path = "/data/lisa/data/faces/GoogleDataset/images/facesCoordinates/"
    save_path = "/data/lisatmp/data/faces_bbox/"
    newsize = [256, 256]
    h5_name = "test_face.h5"
    save_img_data(img_path, bbox_path, save_path, newsize=newsize, h5_name=h5_name, limit=200)

if __name__=="__main__":
    test4()
