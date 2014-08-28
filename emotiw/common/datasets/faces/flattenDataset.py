from faceimages import FaceImagesDataset


class FlattenDataset(FaceImagesDataset):
    """Create a view of a dataset based on bounding boxes"""

    def __init__(self, dataset, bbox_method):
        """
        A mapping between the image index on the corresponding bounding box.
        For example, the index 4 represents the 4th bounding box and
        the bbox_idx_map attribute gives the corresponding image index
        and the informations about the bounding box.

        Parameters
        ----------
        dataset: FaceImagesDataset
            The dataset to flatten.
        bbox_method: str
            The bounding box method to use for flattening
        """

        super(FlattenDataset, self).__init__("FlattenDataset")

        self.dataset = dataset()
        self.bbox_idx_map = []
        self.bbox_method = bbox_method

        for i in range(len(self.dataset)):
            get_bbox = getattr(self.dataset, self.bbox_method)
            bboxes = get_bbox(i)
            if bboxes is not None:
                for bbox in bboxes:
                    self.bbox_idx_map.append((i, [bbox]))
            else:
                self.bbox_idx_map.append((i, None))

    def __len__(self):
        return len(self.bbox_idx_map)

    def get_7emotion_index(self, i):
        return self.dataset.get_7emotion_index(self.bbox_idx_map[i][0])

    def get_7emotion_label(self, i):
        return self.dataset.get_7emotion_label(self.bbox_idx_map[i][0])

    def get_detailed_emotion_label(self, i):
        return self.dataset.get_detailed_emotion_label(self.bbox_idx_map[i][0])

    def get_eyes_location(self, i):
        return self.dataset.get_eyes_location(self.bbox_idx_map[i][0])

    def get_facs(self, i):
        return self.dataset.get_facs(self.bbox_idx_map[i][0])

    def get_gaze_direction(self, i):
        return self.dataset.get_gaze_direction(self.bbox_idx_map[i][0])

    def get_gender(self, i):
        return self.dataset.get_gender(self.bbox_idx_map[i][0])

    def get_head_pose(self, i):
        return self.dataset.get_head_pose(self.bbox_idx_map[i][0])

    def get_is_mouth_opened(self, i):
        return self.dataset.get_is_mouth_opened(self.bbox_idx_map[i][0])

    def get_keypoints_location(self, i):
        return self.dataset.get_keypoints_location(self.bbox_idx_map[i][0])

    def get_light_source_direction(self, i):
        return self.dataset.get_light_source_direction(self.bbox_idx_map[i][0])

    def get_mapped_bbox(self, i):
        return self.bbox_idx_map[i][1]

    def get_name(self):
        return "Flatten dataset"

    def get_original_bbox(self, i):
        bboxes = None

        if self.bbox_method == "get_original_bbox":
            bboxes = self.bbox_idx_map[i][1]
        else:
            bboxes = self.dataset.get_original_bbox(self.bbox_idx_map[i][0])

        return bboxes

    def get_original_image_path(self, i):
        return self.dataset.get_original_image_path(self.bbox_idx_map[i][0])

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.dataset.get_original_image_path_relative_to_base_directory(self.bbox_idx_map[i][0])

    def get_picasa_bbox(self, i):
        bboxes = None

        if self.bbox_method == "get_picasa_bbox":
            bboxes = self.bbox_idx_map[i][1]
        else:
            bboxes = self.dataset.get_picasa_bbox(self.bbox_idx_map[i][0])

        return bboxes

    def get_pyvision_bbox(self, i):
        bboxes = None

        if self.bbox_method == "get_pyvision_bbox":
            bboxes = self.bbox_idx_map[i][1]
        else:
            bboxes = self.dataset.get_pyvision_bbox(self.bbox_idx_map[i][0])

        return bboxes

    def get_ramanan_keypoints_location(self, i):
        return self.dataset.get_ramanan_keypoints_location(self.bbox_idx_map[i][0])

    def get_subject_id_of_ith_face(self, i):
        return self.dataset.get_subject_id_of_ith_face(self.bbox_idx_map[i][0])
