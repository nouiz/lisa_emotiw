import tables
import PIL.Image
from crop_face import crop_face

class ImgStruct(tables.IsDescription):
    idx = tables.Int64Col()
    data = tables.StringCol(96*96*4)

class LabelStruct(tables.IsDescription):
    idx = tables.Int64Col()
    name = tables.Int8Col()
    col = tables.Int64Col()
    row = tables.Int64Col()

def to_hdf5(wrapper, save_as=None):
    if save_as is None:
        save_as = wrapper.dataset_name.replace(' ', '_') + '.h5'

    keypoints_to_id = {'left_eyebrow_inner_end': 1, 'left_eyebrow_outer_end': 27, 'mouth_top_lip_bottom': 2, 'right_ear_canal': 3, 'face_right': 52, 'right_ear_top': 4, 'right_ear_center': 29, 'mouth_top_lip': 5, 'right_eye_center_top': 25, 'mouth_bottom_lip_top': 6, 'right_eyebrow_center': 7, 'face_center': 31, 'chin_left': 8, 'right_eyebrow_center_top': 33, 'face_left': 53, 'left_eyebrow_center': 34, 'right_eye_pupil': 35, 'right_nostril': 42, 'mouth_left_corner': 37, 'left_eye_center_bottom': 38, 'nose_tip': 9, 'nostrils_center': 18, 'left_eyebrow_center_top': 10, 'left_eye_outer_corner': 11, 'mouth_right_corner': 41, 'right_eye_inner_corner': 32, 'left_nostril': 48, 'right_eye_center': 43, 'right_ear': 12, 'left_ear_top': 28, 'mouth_bottom_lip': 13, 'left_eye_center': 14, 'left_mouth_outer_corner': 15, 'right_eyebrow_outer_end': 45, 'left_eye_center_top': 16, 'left_ear_center': 17, 'left_eye_pupil': 46, 'left_eyebrow_center_bottom': 39, 'right_ear_bottom': 36, 'right_eyebrow_inner_end': 26, 'right_eye_center_bottom': 20, 'chin_center': 21, 'left_eye_inner_corner': 22, 'mouth_center': 47, 'right_mouth_outer_corner': 23, 'left_ear_bottom': 24, 'nose_center_top': 30, 'right_eyebrow_center_bottom': 49, 'left_ear_canal': 50, 'right_eye_outer_corner': 19, 'left_ear': 51, 'chin_right': 44}
    
    f = None
 
    try:
        f = tables.openFile(save_as, mode='w')

        train_group = f.createGroup('/', 'train', 'train set')
        test_group = f.createGroup('/', 'test', 'test set')
        train_img_group = f.createGroup('/train', 'data', 'data group')
        train_label_group = f.createGroup('/train', 'label', 'label group')
        test_img_group = f.createGroup('/test', 'data', 'data group')
        test_label_group = f.createGroup('/test', 'label', 'label group')

        img_groups = (train_img_group)
        label_groups = (train_label_group)

        test_idx = None
        dsets[0] = wrapper

        if wrapper.get_standard_train_test_split() is not None:
            dsets[0], dsets[1] = wrapper.get_standard_train_test_split()
            img_groups += (test_img_group)
            label_groups += (test_label_group)
        
        for some_idx in enumerate(img_groups):
            img_group = img_groups[some_idx]
            label_group = label_groups[some_idx] 
            dset = dsets[some_idx]

            img_row = f.createTable(img_group, 'img', ImgStruct, 'image data').row
            label_row = f.createTable(label_group, 'label', LabelStruct, 'target data').row

            for i in dset:
                img, label = crop_face(PIL.Image.open(dset[i].get_original_image_path()),
                                            dset[i].get_bbox(),
                                            dset[i].get_eyes_location(),
                                            dset[i].get_keypoints_location())
                 
                img_row['idx'] = i
                img_row['data'] = img.tostring()
                img_row.append()

                for name, point in label.iteritems():
                    label_row['idx'] = i
                    label_row['name'] = keypoints_to_id[name]
                    label_row['col'] = point[0]
                    label_row['row'] = point[1]
                    label_row.append()

    finally:
        if f is not None:
            f.close()
