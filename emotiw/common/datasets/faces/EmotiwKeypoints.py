from collections import OrderedDict
from pylearn2.utils.string_utils import preprocess
import tables
import os
import numpy
import numpy.random
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter, DenseDesignMatrix
import PIL
import PIL.Image
import pylab
import theano

keypoints_names = ['left_eyebrow_inner_end', 'bottom_lip_top_left_midpoint', 'right_ear_top', 'mouth_bottom_lip_top', 'face_left', 'left_eyebrow_outer_midpoint', 'left_jaw_1', 'left_jaw_0', 'bottom_lip_top_left_center', 'left_eyebrow_center_top', 'left_eye_outer_corner', 'top_lip_bottom_right_midpoint', 'mouth_bottom_lip', 'left_mouth_outer_corner', 'left_eyebrow_center_bottom', 'top_lip_bottom_left_center', 'right_eyebrow_inner_end', 'chin_center', 'right_eyebrow_outer_midpoint', 'left_ear_bottom', 'right_eye_outer_corner', 'left_eyebrow_outer_end', 'top_lip_bottom_left_midpoint', 'bottom_lip_bottom_right_midpoint', 'right_eye_center_top', 'right_nostril_inner_end', 'top_lip_bottom_center', 'face_center', 'right_eye_inner_corner', 'right_eyebrow_center_top', 'left_eyebrow_center', 'right_ear_bottom', 'mouth_left_corner', 'nostrils_center', 'right_eyebrow_inner_midpoint', 'mouth_right_corner', 'chin_center_top', 'nose_ridge_bottom', 'right_eye_center', 'left_eye_bottom_outer_midpoint', 'left_eye_pupil', 'right_jaw_2', 'right_jaw_1', 'right_jaw_0', 'top_lip_bottom_right_center', 'top_lip_top_right_center', 'left_nostril_inner_end', 'right_eyebrow_center_bottom', 'chin_right', 'mouth_top_lip_bottom', 'right_ear_canal', 'bottom_lip_bottom_center', 'mouth_top_lip', 'right_eyebrow_center', 'chin_left', 'left_eye_top_outer_midpoint', 'left_jaw_2', 'nose_tip', 'bottom_lip_bottom_left_center', 'left_eye_top_inner_midpoint', 'right_eye_top_outer_midpoint', 'left_eye_bottom_inner_midpoint', 'top_lip_top_left_center', 'bottom_lip_bottom_right_center', 'bottom_lip_top_center', 'left_eye_center', 'bottom_lip_top_right_midpoint', 'left_eye_center_top', 'left_ear_center', 'top_lip_top_right_midpoint', 'bottom_lip_bottom_left_midpoint', 'right_eye_center_bottom', 'right_eye_bottom_outer_midpoint', 'left_eye_inner_corner', 'right_mouth_outer_corner', 'left_eyebrow_inner_midpoint', 'left_ear_top', 'right_ear_center', 'nose_center_top', 'right_eye_pupil', 'bottom_lip_top_right_center', 'left_eye_center_bottom', 'right_eye_top_inner_midpoint', 'left_cheek_2', 'face_right', 'right_nostril', 'top_lip_top_left_midpoint', 'right_eye_bottom_inner_midpoint', 'left_cheek_1', 'left_cheek_0', 'right_eyebrow_outer_end', 'nose_ridge_top', 'mouth_center', 'left_nostril', 'right_cheek_1', 'right_cheek_0', 'right_cheek_2', 'left_ear_canal']

class EmotiwKeypoints(DenseDesignMatrix):
    def __init__(self, which_set, start=None, stop=None, axes=('b', 0, 1, 'c'), stdev=0.8):
 #       self.translation_dict = OrderedDict({1: 'left_eyebrow_inner_end', 2: 'mouth_top_lip_bottom', 3: 'right_ear_canal', 4: 'right_ear_top', 5: 'mouth_top_lip', 6: 'mouth_bottom_lip_top', 7: 'right_eyebrow_center', 8: 'chin_left', 9: 'nose_tip', 10: 'left_eyebrow_center_top', 11: 'left_eye_outer_corner', 12: 'right_ear', 13: 'mouth_bottom_lip', 14: 'left_eye_center', 15: 'left_mouth_outer_corner', 16: 'left_eye_center_top', 17: 'left_ear_center', 18: 'nostrils_center', 19: 'right_eye_outer_corner', 20: 'right_eye_center_bottom', 21: 'chin_center', 22: 'left_eye_inner_corner', 23: 'right_mouth_outer_corner', 24: 'left_ear_bottom', 25: 'right_eye_center_top', 26: 'right_eyebrow_inner_end', 27: 'left_eyebrow_outer_end', 28: 'left_ear_top', 29: 'right_ear_center', 30: 'nose_center_top', 31: 'face_center', 32: 'right_eye_inner_corner', 33: 'right_eyebrow_center_top', 34: 'left_eyebrow_center', 35: 'right_eye_pupil', 36: 'right_ear_bottom', 37: 'mouth_left_corner', 38: 'left_eye_center_bottom', 39: 'left_eyebrow_center_bottom', 41: 'mouth_right_corner', 42: 'right_nostril', 43: 'right_eye_center', 44: 'chin_right', 45: 'right_eyebrow_outer_end', 46: 'left_eye_pupil', 47: 'mouth_center', 48: 'left_nostril', 49: 'right_eyebrow_center_bottom', 50: 'left_ear_canal', 51: 'left_ear', 52: 'face_right', 53: 'face_left'})
        if which_set not in ('train', 'test'):
            raise ValueError('which_set must be one of ("train", "test")')

        self.stdev = stdev

        self.pixels = numpy.arange(0, 96)
        self.which_set = which_set
        
        X = numpy.memmap(preprocess('${PYLEARN2_DATA_PATH}/faces/hdf5/complete_' + which_set + '_x.npy'))
        Y = numpy.memmap(preprocess('${PYLEARN2_DATA_PATH}/faces/hdf5/complete_' + which_set + '_y.npy'), dtype=numpy.float32)
        
        num_examples = len(X)/(96.0*96.0*3.0)

        if stop is None:
            stop = num_examples
        if start is None:
            start = 0
        
        X = X.view()[start*96*96*3:stop*96*96*3]
        Y = Y.view()[start*len(keypoints_names)*2:stop*len(keypoints_names)*2]
        X.shape = (stop-start, 96*96*3)
        Y.shape = (stop-start, len(keypoints_names), 2)
        Y = self.make_targets(Y)
        
        super(EmotiwKeypoints, self).__init__(X=X, y=Y, view_converter=DefaultViewConverter(shape=[96, 96, 3], axes=axes))

    def has_targets(self):
        return True

    def convert_to_one_hot(self, min_class=0):
        raise NotImplementedError("Keypoints can't be represented as one-hot vectors")

    def get_topo_batch_axis(self):
        self.axes.index('b')

    def adjust_for_viewer(self, X):
        if len(X.shape) == 1:
            return map(lambda x: x/1.3, X)
        else:
            return [map(lambda x: x/1.3, y) for y in X]
 
    def make_targets(self, y):
        #Inspired by Nicholas' FacialKeyoint's make_targets
        Y = numpy.memmap('/tmp/density_vectors.npy', dtype='float32', mode='write', shape=((y.shape[0], y.shape[1], 96, 2)))
        y = y.view()
        self.pixels = self.pixels.view()
        self.pixels.shape = (96, 1)
        y.shape=(y.shape[0], y.shape[1], 1, y.shape[2])

        for i in xrange(y.shape[0]):
            this = numpy.where(y[i,:,:,:] != -1,
                                (numpy.exp(-(y[i,:,:,:]-self.pixels)**2/(2*self.stdev**2)))/(numpy.sqrt(2*3.14159265359)*self.stdev),
                                -1)
            Y[i, :, :] = this

            if i % 10000 == 0 and i != 0:
                Y.flush()

        #Y = Y.view()
        #Y.shape = (y.shape[0], 2*y.shape[1], 96)

        print Y.shape
        return Y

#def test_works():
#    rng = numpy.random.RandomState()
#
#    train_wrapper = HDF5KeypointsWrapper('train')
#    test_wrapper = HDF5KeypointsWrapper('test')
#
#    train_dmat = train_wrapper.get_design_matrix()
#    test_dmat = test_wrapper.get_design_matrix()
#
#    train_targets = train_wrapper.get_targets()
#    test_targets = test_wrapper.get_targets()
#
#    train_wview = train_wrapper.get_weights_view()
#    test_wview = test_wrapper.get_weights_view()
#
#    train_topo_view = train_wrapper.get_topological_view()
#    test_topo_view = test_wrapper.get_topological_view()
#
#    assert len(train_dmat) == sum(train_wrapper.elems_in_files)
#    assert len(test_dmat) == sum(test_wrapper.elems_in_files)
#
#    assert(len(train_dmat) == len(train_targets))
#    assert(len(test_dmat) == len(test_targets))
#
#
#    for i in xrange(1):
#        train_bdesign = train_wrapper.get_batch_design(10)
#        test_bdesign = test_wrapper.get_batch_design(10)
#        
#        assert(len(train_bdesign) == 10)
#        assert(len(test_bdesign) == 10)
#        assert(len(train_bdesign[0]) == 96*96*3)
#        assert(len(test_bdesign[0]) == 96*96*3)
#
#        idx = rng.randint(min(len(train_dmat), len(test_dmat)))
#
#        #NOTE: The following tests are rather slow. Thy may take a few minutes to complete.
#        assert(len(train_wview[idx]) == 96)
#        for y_i, y in enumerate(train_wview[idx]):
#            assert(len(y) == 96)
#            for z_i, z in enumerate(y):
#                assert(len(z) == 3)
#                for w_i, w in enumerate(z):
#                    assert(train_topo_view[idx][y_i][z_i, w_i] == w)
#
#        assert(len(test_wview[idx]) == 96)
#        for y_i, y in enumerate(test_wview[idx]):
#            assert(len(y) == 96)
#            for z_i, z in enumerate(y):
#                assert(len(z) == 3)
#                for w_i, w in enumerate(z):
#                    assert(test_topo_view[idx][y_i][z_i, w_i] == w)
#
#        for x in train_dmat[idx]:
#            assert(isinstance(x, float))
#            assert(-1.3 <= x <= 1.3)
#
#        for x in test_dmat[idx]:
#            assert(isinstance(x, float) and -1.3 <= x <= 1.3)
#
#        #those tests are way too slow, taking easily 8 minutes to complete a single iteration
#        #for the train_wrapper alone.
#
#        #for x in train_wrapper.adjust_for_viewer(train_dmat[idx]):
#         #   for y in x:
#          #      assert(-1 <= y <= 1)
#
#        #for x in train_wrapper.adjust_for_viewer(test_dmat[idx]):
#         #   for y in x:
#          #      assert(-1 <= y <= 1)
#
#    print 'Success!'
#
#if __name__ == '__main__':
#    test_works()
