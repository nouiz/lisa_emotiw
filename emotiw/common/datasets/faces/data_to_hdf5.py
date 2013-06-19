import tables
import PIL.Image
from crop_face import crop_face
from faceimages import FaceDatasetExample, keypoints_names

class ImgStruct(tables.IsDescription):
    data = tables.StringCol(96*96*3)

class LabelStruct(tables.IsDescription):
    name = tables.Int8Col()
    col = tables.Float64Col()
    row = tables.Float64Col()

def to_hdf5(wrapper, save_as=None):
    if save_as is None:
        save_as = wrapper.dataset_name.replace(' ', '_') + '.h5'
    
    f = None

    print wrapper.dataset_name
 
    try:
        f = tables.openFile(save_as, mode='w')

        train_group = f.createGroup('/', 'train', 'train set')
        test_group = f.createGroup('/', 'test', 'test set')

        img_groups = (train_group, None)

        test_idx = None
        dsets = [range(len(wrapper)), None]

        if wrapper.get_standard_train_test_splits() is not None:
            dsets[0], dsets[1] = wrapper.get_standard_train_test_splits()
            img_groups = (img_groups[0], test_group)

        for some_idx, testing in enumerate(dsets):
            if testing is None: 
                break
            img_group = img_groups[some_idx]
            dset = dsets[some_idx]

            img_table = f.createCArray(img_group, 'img', tables.StringAtom(itemsize=1), shape=(len(dset), 96*96*3))
            label_table = f.createCArray(img_group, 'label', tables.Float64Atom(), shape=(len(dset), len(keypoints_names), 2))

            for i, _ in enumerate(dset):
                img, label = crop_face(PIL.Image.open(wrapper.get_original_image_path(i)),
                                            wrapper.get_bbox(i),
                                            wrapper.get_eyes_location(i),
                                            wrapper.get_keypoints_location(i))
                 
                img_table[i, :] = img.tostring()
                for name in keypoints_names:
                    if name in label:
                        point = label[name]
                        label_table[i, keypoints_names.index(name), :] = [point[0], point[1]]
                    else:
                        label_table[i, keypoints_names.index(name), :] = [-1, -1]

    finally:
        if f is not None:
            f.close()
