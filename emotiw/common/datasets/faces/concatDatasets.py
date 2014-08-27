from faceimages import FaceImagesDataset


class ConcatDatasets(FaceImagesDataset):

    def __init__(self, datasets):
        super(ConcatDatasets, self).__init__("ConcatDatasets")

        self.datasets = []

        for dataset in datasets:
            dataset_inst = dataset()
            try:
                getseq = getattr(dataset_inst, 'get_sequence')
                for i in range(len(dataset_inst)):
                    self.datasets.append(getseq(i))
            except AttributeError:
                self.datasets.append(dataset_inst)

    def __len__(self):
        return reduce(lambda x, y: x + y, map(len, self.datasets))

    def get_name(self):
        return "Concat datasets"

    def get_dataset_method(self, i, method):
        dataset_method = None
        prev_len = 0

        for dataset in self.datasets:
            data_len = len(dataset)
            if i <= data_len + prev_len:
                i -= prev_len
                dataset_method = getattr(dataset, method)
                break

            prev_len += data_len

        return dataset_method(i)

    def get_bbox(self, i):
        return self.get_dataset_method(i, 'get_bbox')

    def get_original_bbox(self, i):
        return self.get_dataset_method(i, 'get_original_bbox')

    def get_picasa_bbox(self, i):
        return self.get_dataset_method(i, 'get_picasa_bbox')

    def get_pyvision_bbox(self, i):
        return self.get_dataset_method(i, 'get_pyvision_bbox')

    def get_eyes_location(self, i):
        return self.get_dataset_method(i, 'get_eyes_location')

    def get_keypoints_location(self, i):
        return self.get_dataset_method(i, 'get_keypoints_location')

    def get_ramanan_keypoints_location(self, i):
        return self.get_dataset_method(i, 'get_ramanan_keypoints_location')

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.get_dataset_method(i, 'get_original_image_path_relative_to_base_directory')

    def get_detailed_emotion_label(self, i):
        return self.get_dataset_method(i, 'get_detailed_emotion_label')

    def get_7emotion_index(self, i):
        return self.get_dataset_method(i, 'get_7emotion_index')

    def get_7emotion_label(self, i):
        return self.get_dataset_method(i, 'get_7emotion_label')

    def get_original_image_path(self, i):
        return self.get_dataset_method(i, 'get_original_image_path')

    def get_subject_id_of_ith_face(self, i):
        return self.get_dataset_method(i, 'get_subject_id_of_ith_face')

    def get_facs(self, i):
        return self.get_dataset_method(i, 'get_facs')

    def get_head_pose(self, i):
        return self.get_dataset_method(i, 'get_head_pose')

    def get_light_source_direction(self, i):
        return self.get_dataset_method(i, 'get_light_source_direction')

    def get_gaze_direction(self, i):
        return self.get_dataset_method(i, 'get_gaze_direction')

    def get_gender(self, i):
        return self.get_dataset_method(i, 'get_gender')

    def get_is_mouth_opened(self, i):
        return self.get_dataset_method(i, 'get_is_mouth_opened')
