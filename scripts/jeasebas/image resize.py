from pylab import *

original_dir = "C:\Users\Sebastien\Documents\Faces_Aligned_Test" #May be modified
target_dir = original_dir + "_Small"

import os
import Image

if (os.path.isdir(target_dir) == False):
    os.makedirs(target_dir)

for j in sorted(os.listdir(original_dir)):
    if (os.path.isdir(target_dir+"\\"+j) == False):
        os.makedirs(target_dir+"\\"+j)

for j in sorted(os.listdir(original_dir)):
    for k in sorted(os.listdir(original_dir+"\\"+j)):
        try:
            Image.open(target_dir+"\\"+j+"\\"+k)
        except:
            cur_image = Image.open(original_dir+"\\"+j+"\\"+k)
            mod_image = cur_image.resize((71,90),Image.ANTIALIAS)
            mod_image.save(target_dir+"\\"+j+"\\"+k)
