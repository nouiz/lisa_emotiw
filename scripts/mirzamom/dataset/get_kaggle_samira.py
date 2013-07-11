import glob
import numpy
import ipdb
from skimage import io

path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGL-AFEW/train48/"
data = glob.glob(path + "*.png")
kaggle = []
for item in data:
    if len(item.split('-')) == 2:
        kaggle.append(item)


data_x = []
data_y = []
for item in kaggle:
    arr = numpy.asarray(io.imread(item))
    data_x.append(arr)
    data_y.append(int(item.split('_')[-1].split('.')[0]))

data_x = numpy.concatenate([item[numpy.newaxis,:,:] for item in data_x])
data_y = numpy.asarray(data_y).astype('uint8')

save_path = "/data/lisa/data/faces/EmotiW/preproc/samira/"
numpy.save(save_path + 'data_x.npy', data_x)
numpy.save(save_path + 'data_y.npy', data_y)
