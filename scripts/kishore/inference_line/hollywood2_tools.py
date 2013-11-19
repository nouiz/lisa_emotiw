#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

def cliplist_from_clipset(clipset_filename, avi_dir):
    clipset_file = open(clipset_filename, 'r')
    clip_list = clipset_file.read().splitlines()
    clipset_file.close()
    clip_paths = []
    for line in clip_list:
        line = line.split(' ')
        if line[-1] == '1':
            clip_paths.append('%s/%s.avi' % (avi_dir, line[0]))
    return clip_paths

def get_whole_clipset(clipset_filename, avi_dir):
    clipset_file = open(clipset_filename, 'r')
    clip_list = clipset_file.read().splitlines()
    clipset_file.close()
    clip_paths = []
    for line in clip_list:
        line = line.split(' ')
        clip_paths.append('%s/%s.avi' % (avi_dir, line[0]))
    return clip_paths

# vim: set ts=4 sw=4 sts=4 expandtab:



