#!/bin/bash

SOURCEDIR="/data/lisatmp2/emo_video/new_clips/emotion_dataset"
DESTDIR="/data/lisatmp2/emo_video/new_clips/emotion_dataset_compressed"
INITDIR=$(pwd)

cd $SOURCEDIR

for FNAME in *.avi
do
    echo
    echo '======================================================='
    echo $FNAME
    ffmpeg -i $FNAME -vcodec msmpeg4v2 -b 400kb ${DESTDIR}/${FNAME}
done

cd $INITDIR
