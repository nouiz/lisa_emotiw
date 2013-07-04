# -*- coding: utf8 -*-
import glob
import os

import numpy as np


rng = np.random.RandomState((2013, 06, 27))

users = [
    "Pascal Lamblin",
    "Guillaume Desjardins",
    "Mehdi Mirza",
    "Guillaume Alain",
    "Çaǧlar Gülcehre",
    "Stephan Gouws",
    "Eric Martin",
    "Aaron Courville",
    "Arnaud Bergeron",
    "Li Yao",
    "Hani Almousli",
    "Raul Chandias Ferrari",
    "Heng Luo",
    "Jean-Philippe Raymond",
    "Jeremie Zumer",
    "Nicholas Leonard",
    "Sina Honari",
    "Roland Memisevic",
    "Arjun Sharma",
    "Abhishek Aggarwal",
    "Eric Laufer",
    "Atousa Torabi",
    "Samira Ebrahimi",
    ]

clips_dir = '/data/lisatmp2/emo_video/new_clips/emotion_dataset'

clips = glob.glob(os.path.join(clips_dir, '*.avi'))
clips.sort()

rng.shuffle(clips)
splits = np.array_split(clips, len(users))

for u, s in zip(users, splits):
    print '-' * 72
    print '%s:' % u
    print
    for c in s:
        print os.path.basename(c)
