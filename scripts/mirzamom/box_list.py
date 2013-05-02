"""
This script organize bounding box so that we have one file for each clop
"""



import os
import glob
import numpy
from skimage import transform, io
from gif import sort_nicely
import ipdb

def resize(input, size):

    img = io.imread(input)
    img = transform.resize(img, size)
    return img


def make_face_tubes(input, output):

    if format == 'jpg':
        io.imsave(output, img)
    else:
        output = output.rstrip('jpg') + '.npy'
        numpy.save(output, img)

if __name__ == "__main__":

    src = "/data/lisa/data/faces/EmotiW/picasa_boxes/"
    dest = "/data/lisa/data/faces/EmotiW/picasa_boxes_per_clip/"
    sets = ["Train", "Val"]
    emots = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Disgust"]

    for set_n in sets:
        print set_n
        for emot in emots:
            print emot
            cur_dest = "{}{}/{}/".format(dest, set_n, emot)
            if not os.path.isdir(cur_dest):
                os.makedirs(cur_dest)

            files = glob.glob("{}{}/{}/*.txt".format(src, set_n, emot))
            clips = [item.split("/")[-1].split("-")[0] for item in files]
            clips = numpy.unique(clips)
            for clip in clips:
                outfname = "{}{}/{}/{}.txt".format(dest, set_n, emot, clip)
                frames = glob.glob("{}{}/{}/{}*.txt".format(src, set_n, emot, clip))
                out_str = ""
                for frame in frames:
                    data = numpy.loadtxt(frame, delimiter=',')
                    if data.ndim == 1:
                        data = data.reshape((1, 4))

                    frame_id = frame.split("/")[-1].rstrip('.txt').split("-")[-1]
                    out_str += "{}: ".format(frame_id)
                    for point in data:
                        out_str += "{}, {}, {}, {};".format(point[0], point[1], point[2], point[3])

                    out_str += "\n"
                with open(outfname, 'w') as outf:
                    outf.write(out_str)
