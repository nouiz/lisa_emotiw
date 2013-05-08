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
            clips = [item.split("/")[-1].split("-")[0] for item in files]
            clips = numpy.unique(clips)
            for clip in clips:
                print clip
                tubes = glob.glob("{}{}/{}/{}*.jpg".format(src, set_n, emot, clip))
                tubes = [item.split("/")[-1].split("-")[1] for item in tubes]
                tubes = numpy.unique(tubes)
                for tube in tubes:
                    frames = glob.glob("{}{}/{}/{}-{}*.jpg".format(src, set_n, emot, clip, tube))
                    sort_nicely(frames)
                    rval = []
                    for frame in frames:
                        rval.append(resize(frame, size))

                    rval = numpy.concatenate([item[numpy.newaxis, ...] for item in rval])
                    rval = rval.astype('uint8')
                    output = "{}{}/{}/{}-{}.npy".format(dest, set_n, emot, clip, tube)
                    numpy.save(output, rval)
