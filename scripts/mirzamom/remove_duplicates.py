import glob
import os

src = "/data/lisa/data/faces/EmotiW/mashape_keypoints/Val/Neutral/"
dest = "/home/www-etud/usagers/mirzamom/HTML/data/"

src_list = glob.glob(src + "*.json")
for item in src_list:
    fname = item.rstrip(".josn") + ".png"
    fname = dest + fname.split("/")[-1]
    if os.path.isfile(fname):
        os.remove(fname)
        print fname


