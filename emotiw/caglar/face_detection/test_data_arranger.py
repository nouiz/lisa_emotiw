from pylearn2.utils.iteration import SequentialSubsetIterator

from imagenet import Imagenet
from face_bbox import FaceBBox
from face_existance_table import FaceTable
from data_arranger import DataArranger

def test1():
    path = '/Tmp/gulcehrc/imagenetTemp.h5'
    path2 = '/Tmp/gulcehrc/imagenetTemp2.h5'

    nonface_ds = Imagenet(which_set = 'train',
            path = path2,
            path_org = path,
            size_of_receptive_field=(8, 8),
            center=True,
            scale=True,
            start=0,
            stop=1000,
            imageShape = (256, 256),
            mode='w',
            axes=('b', 0, 1, 'c'),
            preprocessor=None)

    which_set = "train"
    path = "/data/lisatmp/data/faces_bbox/test_face.h5"
    start = 0
    stop = 1000
    size_of_receptive_field = [8, 8]
    stride = 8
    use_output_map = True

    batch_size = 100
    num_batches = 5
    mode = SequentialSubsetIterator

    face_ds = FaceBBox(which_set=which_set,
                        start=start,
                        stop=stop,
                        use_output_map=use_output_map,
                        stride=stride,
                        size_of_receptive_field=size_of_receptive_field,
                        path=path)

    ft = FaceTable(2000, 1000, 1000, is_filled=False)

    data_arranger = DataArranger(face_ds, nonface_ds, face_ratio=0.5, operate_on_patches=False,
            face_table=ft, total_n_exs=2000)

    #import ipdb; ipdb.set_trace()

    iterator = data_arranger.iterator(mode=mode, batch_size=batch_size, targets=True)
    for data in iterator:
        import ipdb; ipdb.set_trace()
        print "===="
        print "Image ", data[0]
        #print "Targets ", data[1]
        print "===="

if __name__=="__main__":
    test1()
