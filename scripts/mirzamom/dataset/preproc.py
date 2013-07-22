from pylearn2.utils import serial

path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/"
data serial.load(path + 'afew_val.pk')
data['data_x']

dditems = Preprocessing.GlobalContrastNormalization()
pipline = Pipeline()
pipline.items.append(Preprocessing.GlobalContrastNormalization())


