import Image
import numpy
from datetime import datetime
import math
import glob
import os
import pickle
import sys
from crop_match import match_subregion



def get_images(original, cropped):
    # Generate the required numpy arrays
    original = Image.open(original)
    cropped = Image.open(cropped)
    #originalArr = numpy.transpose(numpy.array(original), [2, 0, 1]).astype('float32')
    #croppedArr = numpy.transpose(numpy.array(cropped), [2, 0, 1]).astype('float32')
    originalArr = numpy.array(original).astype('float64')
    croppedArr = numpy.array(cropped).astype('float64')
    return originalArr, croppedArr


def save(name, vals):

    with open(name, 'w') as outf:
        for val in vals:
            str = "{}, {}, {}, {}\n".format(int(val[0]), int(val[1]), int(val[2]), int(val[3]))
            outf.write(str)

if __name__ == "__main__":

    orig_path = "/data/lisa/data/faces/EmotiW/images/"
    cropped_path = "/data/lisa/data/faces/EmotiW/picasa_faces/"
    save_path = "/data/lisa/data/faces/EmotiW/picasa_boxes/"
    _, orig_path, cropped_path, save_path = sys.argv
    missed = []
    files = glob.glob("{}/*.png".format(orig_path))
    for orig in files:
        crop = orig.split('/')[-1].rstrip('.png')
        crops = glob.glob("{}/{}*".format(cropped_path, crop))
        corners = []
        for crop in crops:
            if os.path.isfile(crop):
                orig_arr, crop_arr = get_images(orig, crop)
                try:
                    res =  match_subregion(orig_arr, crop_arr) #, order = 1)
                    print "passed: {}".format(crop)
                    res = [res[1], res[0], res[1] + crop_arr.shape[1], res[0] + crop_arr.shape[0]]
                    corners.append(res)
                except:
                    print "failed: {}".format(crop)
                    missed.append([orig])
        if len(corners) > 0:
            print corners
            save_name = "{}/{}.txt".format(save_path ,orig.split('/')[-1].rstrip('.png'))
            save(save_name, corners)



    print "Done, failed on: ", missed
    with open("failed.pkl", 'w') as output:
        pickle.dump(missed, output)
