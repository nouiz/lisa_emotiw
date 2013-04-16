"""
AFEW2 contains the data distributed with the EmotiW challenge.
"""
import glob
import os

import afew


class AFEW2ImageSequenceDataset(afew.AFEWImageSequenceDataset):
    absolute_base_directory = "/data/lisa/data/faces/EmotiW/images"
    picasa_boxes_base_directory = "/data/lisa/data/faces/EmotiW/picasa_boxes"

    def __init__(self):
        self.imagesequences = []
        self.labels = []
        self.trainIndexes = []
        self.validIndexes = []

        # For each split (Train or Val)
        idx = 0
        splits = (("Train", self.trainIndexes),
                  ("Val", self.validIndexes))
        for split_name, split_index in splits:
            print 'processing %s' % split_name
            for emo_name in sorted(self.emotionNames.keys()):
                print '  %s' % emo_name
                # Directory containing the images for all clips of that
                # emotion in that split
                img_dir = os.path.join(self.absolute_base_directory,
                        split_name, emo_name)
                if not os.path.isdir(img_dir):
                    continue

                # Directory containing the picasa bounding boxes for clips
                # of that emotion in that split
                picasa_bbox_dir = os.path.join(
                        self.picasa_boxes_base_directory,
                        split_name, emo_name)

                # Find all image names
                img_names = glob.glob(os.path.join(img_dir, '*.jpg'))
                print '%s img_names' % len(img_names)

                # Find all clips (sequences)
                unique_seq = sorted(set([img.split('-')[0]
                                         for img in img_names]))
                print '%s unique_seq' % len(unique_seq)

                # For each clip
                for seq in unique_seq:
                    # Load the Image Sequence object
                    im_seq = afew.ImageSequence("AFEW2",
                            img_dir, "{0}-*.jpg".format(seq),
                            picasa_bbox_dir, self.emotionNames[emo_name],
                            csv_delimiter=',')
                    self.imagesequences.append(im_seq)

                    # Save label
                    self.labels.append(self.emotionNames[emo_name])

                    # Save split
                    split_index.append(idx)

                    idx += 1

                print '  done, idx = %s' % idx

            print 'done, idx = %s' % idx
        print 'finished, idx = %s' % idx
