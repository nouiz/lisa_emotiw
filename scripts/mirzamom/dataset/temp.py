import glob
import numpy
import ipdb
from skimage import io

def read_data(data):
    data_x = []
    data_y = []
    for item in data:
        arr = numpy.asarray(io.imread(item))
        data_x.append(arr)
        data_y.append(int(item.split('_')[-1].split('.')[0]))

    data_x = numpy.concatenate([item[numpy.newaxis,:,:] for item in data_x])
    data_y = numpy.asarray(data_y).astype('uint8')
    return data_x, data_y



path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGL-AFEW/train48/"
# kaggle
data = glob.glob(path + "*.png")
selected = []
for item in data:
    if len(item.split('-')) == 2:
        selected.append(item)

data_x, data_y = read_data(selected)
save_path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/"
numpy.save(save_path + 'train_kaggle_x.npy', data_x)
numpy.save(save_path + 'train_kaggle_y.npy', data_y)

# challenge
data = glob.glob(path + "*.png")
selected = []
for item in data:
    if len(item.split('-')) !=2:
        selected.append(item)

data_x, data_y = read_data(selected)
save_path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/"
numpy.save(save_path + 'train_afew2_x.npy', data_x)
numpy.save(save_path + 'train_afew2_y.npy', data_y)

# valid
path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGL-AFEW/valid48/"
# challenge
data = glob.glob(path + "*.png")
selected = []
for item in data:
    if len(item.split('-')) !=2:
        selected.append(item)

data_x, data_y = read_data(selected)
save_path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/"
numpy.save(save_path + 'valid_afew2_x.npy', data_x)
numpy.save(save_path + 'valid_afew2_y.npy', data_y)


