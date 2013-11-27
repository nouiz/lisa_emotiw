import os
import sys
import numpy as np

classnames = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def export_label_and_probs(input_npy_fp, output_label_fp, output_prob_fp):
    """Reads probabilities npy file, exports the label and prob in txt files

    input_npy_fp: name of the .npy file
    output_label_fp: name of the file containing the label only
    output_prob_fp: name of the file containing label and probabilities
    """
    probs = np.load(input_npy_fp).flatten()
    assert len(probs) == 7
    label = classnames[probs.argmax()]

    output_label = open(output_label_fp, 'w')
    print >> output_label, label
    output_label.close()

    output_prob = open(output_prob_fp, 'w')
    print >> output_prob, label, ('%f ' * 7) % tuple([p for p in probs])
    output_prob.close()


if __name__ == '__main__':
    pred_dir = sys.argv[1]
    final_pred_dir = sys.argv[2]
    clip_ids = sys.argv[3:]

    for clip_id in clip_ids:
        input_npy_fp = os.path.join(pred_dir, 'xavier_output_%s.npy' % clip_id)
        output_label_fp = os.path.join(final_pred_dir, clip_id)
        output_prob_fp = os.path.join(final_pred_dir, '%s_probabilities.txt' % clip_id)
        export_label_and_probs(input_npy_fp, output_label_fp, output_prob_fp)
