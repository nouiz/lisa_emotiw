from convert_to_h5file import *


if __name__=="__main__":
    img_path = "/data/lisa/data/faces/GoogleDataset/images/"
    bbox_path = "/data/lisa/data/faces/GoogleDataset/images/facesCoordinates/"
    save_path = "/data/lisatmp/data/faces_bbox/"
    newsize = [256, 256]
    h5_name = "face_bbox2.h5"
    save_img_data(img_path, bbox_path, save_path, newsize=newsize, h5_name=h5_name)
