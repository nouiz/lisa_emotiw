# -*- coding: utf8 -*-
"""simple wrapper around Çağlar's audio code to work around file permissions"""
import sys

sys.path.insert(1, '/data/lisa/data/faces/EmotiWTest/best_audio_classifier')
import save_features_pascal_pipeline

if __name__ == '__main__':
    file_dir = sys.argv[1]
    features_dir = sys.argv[2]
    scores_out_dir = sys.argv[3]
    clip_ids = sys.argv[4:]
    video_files = ['%s.avi' % clip_id for clip_id in clip_ids]
    save_features_pascal_pipeline.run_pipeline(
        file_dir=file_dir,
        features_dir=features_dir,
        scores_out_dir=scores_out_dir,
        video_files=video_files)
