#!/bin/bash

# Arguments:
# $1: clip_id, e.g.: 000143240
# $2: model_dir, e.g.: /data/lisa/exp/faces/emotiw_final/Samira
# $3: data_root_dir, e.g.: /u/ebrahims/emotiw_pipeline/test2
# script_dir is extracted automatically from path of this script
script_dir="$( cd "$( dirname "$0" )" && pwd )"
clip_id=$1
model_dir=$2
data_root_dir=$3

echo "running in $script_dir, with clip_id=$clip_id, model_dir=$model_dir, and data_root_dir=$data_root_dir."

path7="'$data_root_dir/module_predictions'"

matlab -nodesktop << EOF       #starts Matlab
warning off

cd $script_dir/SVMCodes
addpath(genpath('./libsvm-3.17'))

learnCaglarFeature($path7, '$clip_id')

exit
EOF
