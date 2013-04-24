
import glob
import numpy
import re
from PIL import Image
from images2gif import writeGif

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def get_frames(path):

    rval = {}
    img_list= glob.glob("{}*.jpg".format(path))
    clips = numpy.unique([item.split('/')[-1].rstrip('.jpg').split('-')[0] for item in img_list])
    for clip in clips:
        list = glob.glob("{}{}-*.jpg".format(path, clip))
        rval[clip] = list

    return rval


def make_gif(frames, save_path):
    for clip in frames.keys():
        imgs = frames[clip]
        if len(imgs) > 110:
            continue
        sort_nicely(imgs)
        path = "{}{}.gif".format(save_path, clip)
        imgs = [Image.open(item) for item in imgs]
        writeGif(path, imgs)


def main():

    frame_path = "/data/lisa/data/faces/EmotiW/picasa_face_tubes_full_images/v2/"
    save_path = "/data/lisa/data/faces/EmotiW/picasa_face_tubes_gifs/v2/"
    emots = ["Angry", "Fear", "Happy", "Disgust", "Neutral", "Sad", "Surprise"]
    #emots = ["Fear", "Happy", "Disgust", "Neutral", "Sad", "Surprise"]
    #emots = ["Angry"]
    sets = ["Train", "Val"]
    stats = {}
    for set_n in sets:
        print set_n
        for emot in emots:
            print emot
            print
            lstat = {}
            frames = get_frames("{}{}/{}/".format(frame_path, set_n, emot))
            make_gif(frames, "{}{}/{}/".format(save_path, set_n, emot))
if __name__ == "__main__":
    main()
