"""
Extracts frame images from avi files
"""

import subprocess
import glob
import os
import sys
import ipdb

def get_output_size(path, width = 1024):
    """
    Read the aspect ration information and return
    an output size string based on aspect ratio
    and given width

    Params
    -----
    path: avi file path
    widht: output image width
    """

    command = ["ffprobe", path]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = p.communicate()[0]
    asr = res[res.find("DAR "):].split(']')[0][4:].split(':')
    try:
        asr = map(float, asr)
    except ValueError:
        asr = res[res.find("DAR "):].split(' ')[1].split(':')
        asr[1] = asr[1].split(",")[0]
        asr = map(float, asr)

    height = int(width / (asr[0]/asr[1]))
    return "{}x{}".format(width, height)


def extract_frames(src, dest, asr):
    """
    Extract frame images from avi

    Params
    -----
    src: src file
    dest: dest file pattern
    asr: aspect ration string e.g. 1024x576
    """

    command = ["ffmpeg", "-i", src,  "-s", asr, "-q", "1", dest]
    subprocess.call(command)

if __name__ == "__main__":

    args = sys.argv
    clip_path = args[1]
    save_path = args[2]
    file_list = args[3:]

    if len(file_list) == 0:
        file_list = glob.glob("{}/*.avi".format(clip_path))
    else:
        for i in xrange(len(file_list)):
            try:
                file_list[i].index('/')
            except ValueError:
                file_list[i] = "{}/{}".format(clip_path, file_list[i])
    print file_list

    # make path if not exist
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for item in file_list:
        # get proper size of image from aspect ration info
        asr = get_output_size(item)

        #extract frame images
        output = "{}/{}-%3d.png".format(save_path, item.split("/")[-1].rstrip(".avi"))
        extract_frames(item, output, asr)

