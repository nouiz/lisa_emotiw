#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import random

import cv
import numpy as np

try:
    import theano
    import theano.tensor as T
    rgbvid = T.ftensor4('rgbvid')
    _convert_to_grayscale = T.cast(T.sum(rgbvid * [[[0.3, 0.59, 0.11]]],
                                   axis=3),
                                   'uint8')
    convert_to_grayscale = theano.function(inputs=[rgbvid],
                                           outputs=_convert_to_grayscale)
except ImportError:
    def convert_to_grayscale(rgb_vid):
        return np.sum(rgb_vid * [[[0.3, 0.59, 0.11]]], axis=3).astype('uint8')


def crop_frame(video, framesize):
    """Crops frames in a video s.t. width and height are multiples of framesize
    """
    return video[:, :np.floor(video.shape[1] / framesize) * framesize,
                 :np.floor(video.shape[2] / framesize) * framesize]

def get_block_from_vid(video, t, y, x, framesize, horizon):
    return video[t:t+horizon, y:y+framesize, x:x+framesize].flatten()

def load_video_clip(video_file, start_frame = 0, end_frame = None, verbose = False):
    """Loads frames from a video_clip

    Args:
        video_file: path of the video file
        start_frame: first frame to be loaded
        end_frame: last frame to be loaded

    Returns:
        A (#frames)x(height)x(width)x(#channels) NumPy array containing the
        video clip
    """
    if not os.path.exists(video_file):
        raise IOError, 'File "%s" does not exist!' % video_file
    capture = cv.CaptureFromFile(video_file)
    if not end_frame:
        end_frame = int(cv.GetCaptureProperty(capture,
                                              cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        end_frame = int(min(end_frame,
                            cv.GetCaptureProperty(capture,
                                                  cv.CV_CAP_PROP_FRAME_COUNT)))
    width = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)
    if verbose:
        print "end_frame: %d" % end_frame
        print "clip has %d frames" % int(
            cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
    for _ in range(start_frame): # frames start with 1 in annotation files
        cv.GrabFrame(capture)
    frames = np.zeros((end_frame - start_frame -2, height, width), dtype=np.uint8)
    for i in range(end_frame - start_frame - 2): # end_frame = last action frame
        img = cv.QueryFrame(capture)
        if img is None:
            continue
        tmp = cv.CreateImage(cv.GetSize(img), 8, 1)
        cv.CvtColor(img, tmp, cv.CV_BGR2GRAY)
        frames[i, :] = np.asarray(cv.GetMat(tmp))
    return np.array(frames)

def sample_clips_dense(video, framesize, horizon, temporal_subsampling, stride=None):
    """Gets dense samples from a video with optional overlap of 50%
    Args:
        video: the video as filepath string or numpy array
        framesize: width (=height) of one sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
        overlap: whether to let the samples overlap 50%
    """
    if type(video) == str:
        if temporal_subsampling:
            video = load_video_clip(video_file=video)[::2]
        else:
            video = load_video_clip(video_file=video)            
    elif temporal_subsampling:
        video = video[::2]

    if  video.shape[0] <= horizon:
            print 'video extended by self concatenation'
            video = np.concatenate((video,video),0)

    if stride!=None:
        nblocks_t = (video.shape[0]-horizon) / (stride[1]) +1
        nblocks_w = (video.shape[2]-framesize) / (stride[0]) +1 
        nblocks_h = (video.shape[1]-framesize) / (stride[0]) +1
        #print nblocks_t, nblocks_w, nblocks_h, nblocks_t*nblocks_w*nblocks_h
        samples = np.zeros((nblocks_t*nblocks_w*nblocks_h, framesize*framesize*horizon), dtype=np.uint8)
        idx = 0
        for frame_idx in range(nblocks_t):
            for col_idx in range(nblocks_w):
                for row_idx in range(nblocks_h):
                    samples[idx, :] = video[frame_idx*stride[1]:frame_idx*stride[1]+horizon,
                                            row_idx*stride[0]:row_idx*stride[0]+framesize,
                                            col_idx*stride[0]:col_idx*stride[0]+framesize
                                            ].flatten()
                    idx += 1
        return samples
    else:
        video = crop_frame(video, framesize)
        video = video[:np.floor(video.shape[0]/horizon)*horizon]
        video = video.reshape((-1, horizon, video.shape[1]/framesize,
                            framesize, video.shape[2]/framesize, framesize))
        video = video.transpose(0, 2, 4, 1, 3, 5).reshape((-1, horizon*framesize**2))
        return video

def sample_clips_random(video, framesize, horizon, temporal_subsampling, nsamples):
    """Gets random samples from one video
    Args:
        video: video filename or numpy array
        framesize: width (=height) of the frames in the sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
        nsamples: number of samples to extract
    Returns:
        NumPy array of shape (nsamples, horizon*framesize*framesize)
    """
    if type(video) == str:
        if temporal_subsampling:
            # take only every 2nd frame
            video = load_video_clip(video_file=video)[::2]
        else:
            video = load_video_clip(video_file=video)
    elif temporal_subsampling:
        video = video[::2]

    if  video.shape[0] <= horizon:
            print 'video extended by self concatenation'
            video = np.concatenate((video,video),0)

    video = crop_frame(video, framesize)
    print video.shape
    #video = convert_to_grayscale(video)
    block_indices = [(np.random.randint(video.shape[0]-horizon), np.random.randint(video.shape[1]-framesize), np.random.randint(video.shape[2]-framesize)) for _ in range(nsamples)]
    return np.vstack([video[frame:frame + horizon, row:row + framesize, col:col + framesize].reshape(1, -1) for (frame, row, col) in block_indices]).astype(np.float32)

def sample_clips_random_from_multiple_videos(videolist, framesize, horizon, temporal_subsampling, nsamples):
    """Gets random samples from multiple videos
    Args:
        videolist: list of video filenames
        framesize: width (=height) of the frames in the sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
        nsamples: number of samples to extract
    Returns:
        NumPy array of shape (nsamples, horizon*framesize*framesize)
    """
    random.shuffle(videolist)
    # if we have more videos than the number of requested samples, we can't
    # sample from all videos
    if len(videolist) > nsamples:
        videolist = videolist[:nsamples]
    nsamples_per_clip = nsamples / len(videolist)


    samples = np.zeros((nsamples, horizon*framesize*framesize), dtype=np.uint8)
    for i in range(len(videolist)):
        print 'sampling from video %d of %d' % (i+1, len(videolist))
        samples[i*nsamples_per_clip:(i+1)*nsamples_per_clip] = \
                sample_clips_random(videolist[i],  framesize, horizon,
                                    temporal_subsampling, nsamples_per_clip)
    if nsamples % len(videolist):
        offset = (nsamples / len(videolist)) * len(videolist)
        samples[offset:] = \
                sample_clips_random(videolist[np.random.randint(len(videolist))], framesize, horizon,
                                    temporal_subsampling, nsamples - offset)
    return np.vstack(samples)

def sample_clips_dense_from_multiple_videos(videolist, framesize, horizon, temporal_subsampling,stride=None):
    """Gets dense samples from multiple videos
    Args:
        videolist: list of video filenames
        framesize: width (=height) of the frames in the sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
    Returns:
        NumPy array of shape (<nsamples>, <number of blocks in vid>, horizon*framesize*framesize)
    """
    samples = []
    for i in range(len(videolist)):
        #print 'sampling from video %d of %d' % (i+1, len(videolist))
        samples.append(
            sample_clips_dense(videolist[i],  framesize, horizon,
                                temporal_subsampling,stride))
    return np.vstack(samples)

if __name__ == '__main__':
    vid = load_video_clip('/home/vincent/data/hollywood2_isa/AVIClips05/actioncliptrain00775.avi')
    horizon = 10
    framesize = 16
    nblocks_t = vid.shape[0]/horizon
    nblocks_w = vid.shape[2]/framesize
    nblocks_h = vid.shape[1]/framesize
    print 'shape of vid: %s (t,y,x)' % (vid.shape, )
    print 'nblocks_t: %d' % (nblocks_t, )
    print 'nblocks_w: %d' % (nblocks_w, )
    print 'nblocks_h: %d' % (nblocks_h, )
    samples = sample_clips_dense(vid, framesize, horizon, temporal_subsampling=False, overlap=True)
    print 'number of samples: %s' % (samples.shape[0], )
    import new_disptools
    new_disptools.create_video_from_patches('/home/vincent/tmp/test_densesampling', samples[:], framesize, vid.shape[1] / (framesize/2), vid.shape[2] / (framesize/2))


# vim: set ts=4 sw=4 sts=4 expandtab:
