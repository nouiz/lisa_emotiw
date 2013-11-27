import numpy as np

predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/xavier_output_%s.npy"
ids = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_filelist.txt"
labels_template = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_targets.npy"

sets = ['valid','train'] # not test because we have no labels!

CLIPS = []
# automatically load ids from file
CLIPS = [line.strip("\n").split("/")[-1] for line in open(ids % 'valid').readlines()]

predictions = []
labels = []

skipped_clips = []

for clip in CLIPS:
    try:
        predictions.append(np.load(predictions_template % clip).argmax())
    except IOError:
        skipped_clips.append(clip)
        continue
    # find position of clip id and type of set (valid or train)
    position = -1
    for s in sets:
        order_file = open(ids % s,'r')
        for line in order_file.readlines():
            line = line.strip("\n")
            if line.split("/")[-1]==clip:
                position = int(line.split(" ")[0])
                break

        if position != -1:
            break

    # load the valid or train file and select label at good position
    labels.append(np.load(labels_template % s)[position])

predictions = np.array(predictions)
labels = np.array(labels)

print "skipped clips"
print skipped_clips
print "mean over %d clips" % predictions.shape[0]
print np.mean(predictions==labels)*100
