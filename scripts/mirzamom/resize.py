import os
import glob
from skimage import transform, io

def resize(input, output, size):

    img = io.imread(input)
    img = transform.resize(img, size)
    io.imsave(output, img)

if __name__ == "__main__":

    src = "/data/lisa/data/faces/EmotiW/picasa_face_tubes/v1/"
    dest = "/data/lisa/data/faces/EmotiW/picasa_face_tubes_96_96/"
    sets = ["Train", "Val"]
    emots = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Disgust"]
    size = (96, 96)

    for set_n in sets:
        print set_n
        for emot in emots:
            print emot
            cur_dest = "{}{}/{}/".format(dest, set_n, emot)
            if not os.path.isdir(cur_dest):
                os.makedirs(cur_dest)

            files = glob.glob("{}{}/{}/*.jpg".format(src, set_n, emot))
            for input in files:
                output = "{}{}".format(cur_dest, input.split("/")[-1])
                resize(input, output, size)
