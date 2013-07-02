from pylearn2.datasets import preprocessing
from face_bbox import FaceBBox

which_set = "train"
path = "/data/lisatmp/data/faces_bbox/test_face_lvl1.h5"
start = 0
stop = 2100
size_of_receptive_field = [128, 128]
stride = 8
use_output_map = True

# Load train data
train = FaceBBox(which_set=which_set,
                        mode="r+",
                        use_output_map=use_output_map,
                        stride=stride,
                        size_of_receptive_field=size_of_receptive_field,
                        path=path)


# prepare preprocessing
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalizationPyTables())
pipeline.items.append(preprocessing.LeCunLCN((256, 256), channels=[0], kernel_size=7))

# apply preprocessing to train
train.apply_preprocessor(pipeline, can_fit = True)
import ipdb; ipdb.set_trace()
