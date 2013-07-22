import glob
import numpy
import ipdb
from skimage import io
from pylearn2.utils import serial
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.datasets.preprocessing import lecun_lcn

#path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGL-AFEW/valid48/"
#path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGL-AFEW/train48/"
path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGLIS-AFEWIS/valid48/"
#path = "/data/lisatmp2/yaoli/datasets/from_pierre/KGLIS-AFEWIS/train48/"

data = glob.glob(path + "*.png")
clean_data = []
for item in data:
    if len(item.split('-')) != 2:
        clean_data.append(item)

data = clean_data
data_x = []
data_y = []
clip_ids = []

for item in clean_data:
    clip_ids.append(item.split('/')[-1].split('_')[1].split('-')[0])

clip_ids = numpy.unique(clip_ids)

for clip in clip_ids:
    arr = []
    for item in data:
        if str(clip) in item:
            arr.append(numpy.asarray(io.imread(item)).reshape((48,48,1)))
            target = int(item.split('/')[-1].split('_')[0])
    data_x.append(numpy.concatenate([item[numpy.newaxis,:,:] for item in arr]))
    data_y.append(target)


data_y = numpy.asarray(data_y).astype('uint8')
clip_ids = numpy.asarray(clip_ids)


assert len(data_x) == len(clip_ids)
assert len(data_x) == len(data_y)


def apply_lcn(data, img_shape, kernel_size = 7, batch_size = 5000):


    data_size = data.shape[0]
    last = numpy.floor(data_size / float(batch_size)) * batch_size
    for i in xrange(0, data_size, batch_size):
        stop = i + numpy.mod(data_size, batch_size) if i >= last else i + batch_size
        data[i:stop, :,:,0] = lecun_lcn(data[i:stop,:,:,0].astype('float32'), img_shape, kernel_size)


    return data


preprocess = True
if preprocess:
    print "Pre-processing the data"
    features = []
    labels = []
    for item, y in zip(data_x, data_y):
        data_shape = item.shape
        item = item / 255.
        item = global_contrast_normalize(item.reshape((data_shape[0], 48*48))).reshape(data_shape)
        item = apply_lcn(item, img_shape = [48,48], kernel_size=5)
        features.append(item.astype('float32'))
        labels.append(y)

    print "Done pre-preprocessing"

data = {'data_x' : features, 'data_y' : labels, 'clip_ids' : clip_ids}

#save_path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/"
save_path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGLIS-AFEWIS/"
serial.save(save_path + 'afew2_valid_prep.pkl', data)
#serial.save(save_path + 'afew2_train_prep.pkl', data)

