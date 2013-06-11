from collections import OrderedDict
import pylearn2.utils.string_utils
import tables
import os
import numpy
import numpy.random
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter, DenseDesignMatrix
import PIL
import PIL.Image
import pylab
import theano

def overlay_me(data, label):
    img = PIL.Image.fromstring(data=data, mode='RGB', size=(96,96))
    x = numpy.argmax(label[0])
    y = numpy.argmax(label[1])
    print (x, y)
    img.putpixel((x, y), (0, 0, 0))
    img.show()

class LazyDesignMatrix(object):
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.shape = (sum(self.wrapper.elems_in_files), 96*96*3) #(num_batches, num_pixels_times_ppc)
        self.transform = lambda x: x

    def get_lazy_topo(self, view_converter):
        lazy_mat = LazyDesignMatrix(self.wrapper)
        lazy_mat.shape = (sum(self.wrapper.elems_in_files), 96, 96, 3)
        lazy_mat.transform = view_converter.design_mat_to_topo_view
        return lazy_mat        

    def __len__(self):
        return sum(self.wrapper.elems_in_files)

    def __getitem__(self, key):
        rept_slice = slice(0, 1, 1)
        the_slice = slice(None, None, None)
        size = sum(self.wrapper.elems_in_files)

        res = []

        if isinstance(key, slice):
            rept_slice = key
        elif isinstance(key, int):
            rept_slice = slice(key, key+1, 1)
        elif isinstance(key, tuple): #A tuple that slices elems first, and data second.
            rept_slice = key[0]
            the_slice = key[1]

        total_examples = 0

        numbers = range(size)[rept_slice]
        if isinstance(numbers, int):
            numbers = [numbers]

        for i in numbers: 
            total_examples += 1
            prev_size = 0
            for idx, size in enumerate(self.wrapper.elems_in_files):
                if size + prev_size > i:
                    the_elem = []
                    map(the_elem.extend, self.wrapper.load_from_hdf5(self.wrapper.file_list[idx], self.wrapper.which_set, i-prev_size, self.wrapper.start, self.wrapper.stop)[the_slice])
                    res.append(the_elem)
                    break
                else:
                    prev_size += size

        if len(res) == 1:
            res = res[0]

        arr = numpy.asarray(res)
        if len(arr.shape) < 2:
            arr = arr.reshape(1, arr.shape[0])
        return self.transform(arr)

class LazyTargets(object):
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.shape = (sum(self.wrapper.elems_in_files), 52) #(num_batches, num_keypoints)

    def __len__(self):
        return sum(self.wrapper.elems_in_files)

    def __getitem__(self, key):
        rept_slice = slice(0, 1, 1)
        the_slice = slice(None, None, None)
        size = sum(self.wrapper.elems_in_files)

        res = []

        if isinstance(key, slice):
            rept_slice = key

        if isinstance(key, tuple): #A tuple that slices elems first, and data second.
            rept_slice = key[0]
            the_slice = key[1]

        for i in range(size)[rept_slice]: 
            prev_size = 0
            for idx, size in enumerate(self.wrapper.elems_in_files):
                if size + prev_size > i:
                    res.append(self.wrapper.load_from_hdf5(self.wrapper.file_list[idx], self.wrapper.which_set, i-prev_size, self.wrapper.start, self.wrapper.stop, False)[the_slice])
                    break
                else:
                    prev_size += size

        return numpy.asarray(res)

class HDF5KeypointsWrapper(DenseDesignMatrix):
    def __init__(self, which_set, start=None, stop=None, axes=('b', 0, 1, 'c'), stdev=0.8):
        self.translation_dict = OrderedDict({1: 'left_eyebrow_inner_end', 2: 'mouth_top_lip_bottom', 3: 'right_ear_canal', 4: 'right_ear_top', 5: 'mouth_top_lip', 6: 'mouth_bottom_lip_top', 7: 'right_eyebrow_center', 8: 'chin_left', 9: 'nose_tip', 10: 'left_eyebrow_center_top', 11: 'left_eye_outer_corner', 12: 'right_ear', 13: 'mouth_bottom_lip', 14: 'left_eye_center', 15: 'left_mouth_outer_corner', 16: 'left_eye_center_top', 17: 'left_ear_center', 18: 'nostrils_center', 19: 'right_eye_outer_corner', 20: 'right_eye_center_bottom', 21: 'chin_center', 22: 'left_eye_inner_corner', 23: 'right_mouth_outer_corner', 24: 'left_ear_bottom', 25: 'right_eye_center_top', 26: 'right_eyebrow_inner_end', 27: 'left_eyebrow_outer_end', 28: 'left_ear_top', 29: 'right_ear_center', 30: 'nose_center_top', 31: 'face_center', 32: 'right_eye_inner_corner', 33: 'right_eyebrow_center_top', 34: 'left_eyebrow_center', 35: 'right_eye_pupil', 36: 'right_ear_bottom', 37: 'mouth_left_corner', 38: 'left_eye_center_bottom', 39: 'left_eyebrow_center_bottom', 41: 'mouth_right_corner', 42: 'right_nostril', 43: 'right_eye_center', 44: 'chin_right', 45: 'right_eyebrow_outer_end', 46: 'left_eye_pupil', 47: 'mouth_center', 48: 'left_nostril', 49: 'right_eyebrow_center_bottom', 50: 'left_ear_canal', 51: 'left_ear', 52: 'face_right', 53: 'face_left'})
        if which_set not in ('train', 'test'):
            raise ValueError('which_set must be one of ("train", "test")')

        self.stdev = stdev

        self.pixels = numpy.arange(0, 96)
        self.which_set = which_set
        self.start = start
        self.stop = stop

        files = ['multipie.h5', 'afw.h5', 'arface.h5', 'aflw.h5', 'ncku.h5', 'hiit6.h5', 'ihdp.h5', 'bioid.h5', 'lfpw.h5', 'caltech.h5', 'inrialpes.h5']
        self.files = [pylearn2.utils.string_utils.preprocess(os.path.join('${KEYPOINTS_DATA_PATH}', 'hdf5_datasets', f)) for f in files]
        self.file_list = [tables.openFile(f) for f in self.files] 
        #TODO: Never actually closed.
    
        self.elems_in_files = [0]*len(files)
        for idx, f in enumerate(self.file_list):
            if len(f.root.test.data._f_listNodes()) != 0:
                self.elems_in_files[idx] = len(f.root.test.data.img) 
            
        super(HDF5KeypointsWrapper, self).__init__(X=[0]*sum(self.elems_in_files), y=[0]*sum(self.elems_in_files), #Although it's OK to have X and Y not actually be features and targets respectively, 
                                                                                                                        #they still have to have the right shape[0].
                                                        view_converter=DefaultViewConverter(shape=[96, 96, 3], axes=axes)) 

    def has_targets(self):
        return True

    def restrict(self, start, stop):
        if stop < start or start < 0 or stop > sum(self.elems_in_files):
            raise ValueError("(%d, %d) is not a valid range. Valid range: (%d, %d)" % (start, stop, 0, sum(self.elems_in_files)))
        if isinstance(start, int):
            self.start = start
        if isinstance(stop, int):
            self.stop = stop

    def convert_to_one_hot(self, min_class=0):
        raise NotImplementedError("Keypoints can't be represented as one-hot vectors")

    def get_topological_view(self, mat=None):
        if mat is None:
            return self.get_design_matrix().get_lazy_topo(self.view_converter)
        #NOTE: might need to create a custom converter
        #so it's aware of the lazy structures.
        return self.view_converter.design_mat_to_topo_view(mat)

    def get_weights_view(self, mat=None):
        return self.get_topological_view(mat)

    def get_batch_design(self, batch_size, include_labels=False):
        #slight adaptation from DenseDesignMatrix
        size = sum(self.elems_in_files)
        the_X = self.get_design_matrix()
        the_y = self.get_targets()

        try:
            idx = self.rng.randint(size - batch_size + 1)
        except ValueError:
            if batch_size > size:
                raise ValueError("Requested "+str(batch_size)+" examples"
                    "from a dataset containing only "+str(size))
            raise
        rx = self.adjust_for_viewer(the_X[idx:idx + batch_size, :])
        if include_labels:
            if the_y is None:
                return rx, None
            ry = the_y[idx:idx + batch_size]
            return rx, ry
        rx = numpy.cast[theano.config.floatX](rx)
        return rx
        
    def get_topo_batch_axis(self):
        return 0

    def adjust_for_viewer(self, X):
        #NOTE: numpy converts '\0' characters in lists of characters as the empty string.
        if len(X.shape) == 1:
            return map(lambda x: len(x) == 1 and (ord(x) - 127.5)/127.5 or 0, X)
        else:
            return [map(lambda x: len(x) == 1 and (ord(x) - 127.5)/127.5 or 0, y) for y in X]
 
    def make_targets(self, y):
        y = numpy.asarray(y)
        #copied straight from FacialKeypoint - only difference is 98->96
        # y : (batch_size, num_keypoints):
        # (batch_size, num_keypoints*2, 96)
        Y = numpy.zeros((y.shape[0], y.shape[1]*2, 96))
        for i in xrange(y.shape[0]):
            for j in xrange(y.shape[1]):
                for k in self.pixels:
                    if y[i, j, 0] == -1:
                        Y[i, j*2, k] = -1
                        Y[i, j*2+1, k] = -1
                    else:
                        Y[i, j*2, k] = numpy.exp((-(y[i, j][0]-k)**2)/(2*(self.stdev**2)))/(numpy.sqrt(2*3.14159265359)*self.stdev)
                        Y[i, j*2+1, k] = numpy.exp((-(y[i, j][1]-k)**2)/(2*(self.stdev**2)))/(numpy.sqrt(2*3.14159265359)*self.stdev)

        print Y.shape
        return Y

    def get_design_matrix(self, topo=None):
        if topo is not None:
            return self.view_converter.topo_view_to_design_mat(topo)
        return LazyDesignMatrix(self)

    def get_targets(self):
        return LazyTargets(self)

    def load_from_hdf5(self, f, which_set, idx, start, stop, data=True):
        #data: whether to load data or targets

        if start is not None:
            assert(idx >= start)
        if stop is not None:
            assert(idx <= stop)

        the_file = f

        img_path = None
        label_path = None

        if which_set == 'train':
            img_path = the_file.root.train.data.img
            label_path = the_file.root.train.label.label
        else:
            img_path = the_file.root.test.data.img
            label_path = the_file.root.test.label.label

        if data:
            datum = img_path[idx]
            lst = datum['data']
            return [lst[i*3:(i+1)*3] for i in xrange(96*96)]
            #return [datum['data'][i*96:(i+1)*96] for i in xrange(96)]

        else:
            labels_dict = {}
            #XXX: VERY slow, shouldn't be needed. Might require recreating the .hdf5
            #to get data in 'O(1)', though.
            for label in label_path:
                if label['idx'] == idx:
                    labels_dict[label['name']] = (label['col'], label['row'])

            labels_dict = OrderedDict(labels_dict)

            labels = []
            for j in self.translation_dict:
                if j in labels_dict:
                    labels.append(numpy.asarray(list(labels_dict[j])))
                else:
                    labels.append(numpy.asarray([-1, -1]))

            return labels

def test_works():
    rng = numpy.random.RandomState()

    train_wrapper = HDF5KeypointsWrapper('train')
    test_wrapper = HDF5KeypointsWrapper('test')

    train_dmat = train_wrapper.get_design_matrix()
    test_dmat = test_wrapper.get_design_matrix()

    train_targets = train_wrapper.get_targets()
    test_targets = test_wrapper.get_targets()

    train_wview = train_wrapper.get_weights_view()
    test_wview = test_wrapper.get_weights_view()

    train_topo_view = train_wrapper.get_topological_view()
    test_topo_view = test_wrapper.get_topological_view()

    assert len(train_dmat) == sum(train_wrapper.elems_in_files)
    assert len(test_dmat) == sum(test_wrapper.elems_in_files)

    assert(len(train_dmat) == len(train_targets))
    assert(len(test_dmat) == len(test_targets))


    for i in xrange(1):
        train_bdesign = train_wrapper.get_batch_design(10)
        test_bdesign = test_wrapper.get_batch_design(10)
        
        assert(len(train_bdesign) == 10)
        assert(len(test_bdesign) == 10)
        assert(len(train_bdesign[0]) == 96*96*3)
        assert(len(test_bdesign[0]) == 96*96*3)

        idx = rng.randint(min(len(train_dmat), len(test_dmat)))

        for y_i, y in enumerate(train_wview[idx]):
            assert(len(y) == 96)
            for x_i, x in enumerate(y):
                assert(len(x) == 96)
                for z_i, z in enumerate(x):
                    assert(len(z) == 3)
                    for w_i, w in enumerate(z):
                        assert(train_topo_view[idx][y_i, x_i][z_i, w_i] == w)

        for y_i, y in enumerate(test_wview[idx]):
            assert(len(y) == 96)
            for x_i, x in enumerate(y):
                assert(len(x) == 96)
                for z_i, z in enumerate(x):
                    assert(len(z) == 3)
                    for w_i, w in enumerate(z):
                        assert(test_topo_view[idx][y_i, x_i][z_i, w_i] == w)

        #those tests are way too slow, taking easily 8 minutes to complete a single iteration of
        #for the train_wrapper alone.
        #for x in train_wrapper.adjust_for_viewer(train_dmat[idx]):
         #   for y in x:
          #      assert(-1 <= y <= 1)

        #for x in train_wrapper.adjust_for_viewer(test_dmat[idx]):
         #   for y in x:
          #      assert(-1 <= y <= 1)

    print 'Success!'

if __name__ == '__main__':
    test_works()
