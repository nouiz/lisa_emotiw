import os
import sys

import numpy as np
from scipy import io as sio

### OPTIONS

DATA_ROOT_DIR = '/u/ebrahims/emotiw_pipeline/EmotiW2014_val'
SAVE_CONF_MAT = 1

# Order of the models
models = [
    "Activity",         # 0. - numpy or txt
    "Audio",            # 1. - numpy
    "Bag of mouth",     # 2. - numpy
    "ConvNet",          # 3. - matlab probs must be under ['prob_values']
    "ConvNet+Audio",    # 4. - matlab probs must be under ['prob_values']
    "Final",            # 5. - numpy
]

USE_MODELS = [
    0,
    1,
#    2,  ## No BoMF for the moment
    3,
    4,
#    5,  ## No final prediction for the moment
]


classnames = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Map the order of the predictions in the file to the order of classnames
# Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
mat_orders = [
    range(7),
    range(7),
    range(7),
    [5, 6, 0, 4, 1, 3, 2],
    [0, 1, 2, 3, 5, 6, 4],
    range(7),
]


PREDICTION_DIR = 'module_predictions'
PREDICTION_DIR = os.path.join(DATA_ROOT_DIR, PREDICTION_DIR)

# Template for the file name of each model's prediction.
# %s is the place-holder for clip_id.
model_pred_templates = [
    'kishore_pred_%s.txt',
    'audio_pred_%s.npy',
    'bomf_pred_%s.npy',
    'svm_convnet_pred_%s.mat',
    'svm_convnet_audio_pred_%s.mat',
    'xavier_output_%s.npy',
]

TARGET_DIR = 'Labels'
TARGET_DIR = os.path.join(DATA_ROOT_DIR, TARGET_DIR)

# The file containing the target has no extension
target_template = '%s'


def load_pred(path, model_idx):
    """Load and return the prediction of the most likely label"""
    if path.split('.')[-1] == 'npy':
        preds = np.load(path)
    elif path.split('.')[-1] == 'mat':
        preds = sio.loadmat(path)['prob_values_test']
    elif path.split('.')[-1] == 'txt':
        preds = []
        with open(path, 'r') as txt_file:
            for line in txt_file.readlines():
                # The 7 probabilities should be at the end of the line
                preds.append(
                    [float(prob) for prob in line.strip().split()[-7:]])
        preds = np.array(preds)
    else:
        raise ValueError(
            "the file must be .npy (numpy), .txt or .mat (matlab)")

    if np.prod(preds.shape) == 7:
        preds = preds.reshape(7)

    # Reorder according to model
    preds = preds[mat_orders[model_idx]]

    return np.argmax(preds)


def main(clip_ids):
    ## Load targets as indices into classnames
    targets = []
    label_file_template = os.path.join(TARGET_DIR, target_template)
    for clip_id in clip_ids:
        with open(label_file_template % clip_id, 'r') as label_file:
            targets.append(classnames.index(label_file.next()))

    ## For each selected model, compare model output against target
    for model_idx in USE_MODELS:
        # Unnormalized confusion matrix
        # Each row corresponds to actual labels,
        # each col corresponds to predictions.
        conf_mat = np.zeros((7, 7), dtype='int32')
        pred_template = os.path.join(PREDICTION_DIR,
                                     model_pred_templates[model_idx])

        for i, clip_id in enumerate(clip_ids):
            pred = load_pred(pred_template % clip_id, model_idx)
            conf_mat[targets[i], pred] += 1

        print 'for model %i: %s' % (model_idx, models[model_idx])
        print 'confusion matrix:'
        print conf_mat

        out_file = pred_template.split('.')[0] % 'confusion_matrix' + '.txt'
        if SAVE_CONF_MAT:
            #print 'np.savetxt(%s, conf_mat)' % out_file
            np.savetxt(out_file, conf_mat)

        accuracy = float(np.diag(conf_mat).sum()) / len(clip_ids)
        print 'accuracy: %2.2f%%' % (accuracy * 100)


if __name__ == '__main__':
    clip_ids = sys.argv[1:]
    main(clip_ids)
