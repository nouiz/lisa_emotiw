Steps:

1. Run frame_extractor.py to extract all the frames from the clips
2. Use picasa to extract the mini faces from the frames
3. Run find_match.py to extract the bounding boxes of these mini-faces. It’s basically top left and bottom right coordinates of the mini-face on the frame.
4. Run face_tube_orig.py to copy the mini-faces into new folder and relabel the the files such that the mini-faces are contiguous.