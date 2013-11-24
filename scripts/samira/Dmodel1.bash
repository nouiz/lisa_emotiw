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


# Ramanan step
# inputs = facetubes_path, ramanan_path, avg_pts_path, preprocessed_path
# testvideo/ Ramanan/testvideo/ avgpoint.mat


path1="'$data_root_dir/facetubes_96x96/$1/'"
path2="'$data_root_dir/Ramanan/$1/'"
path3="'$data_root_dir/Registeration/$1/'"
path4="'$data_root_dir/IS/$1/'"
path5="$data_root_dir/IS/$1"
path6="$data_root_dir/batches/$1"
path7="'$data_root_dir/SVM/$1/'"

echo "path1=$path1"
echo "path2=$path2"
echo "path3=$path3"
echo "path4=$path4"
echo "path5=$path5"
echo "path6=$path6"
echo "path7=$path7"


matlab -nodesktop << EOF       #starts Matlab
warning off
%mkdir $data_root_dir/Ramanan
%mkdir $data_root_dir/Ramanan/$1
%cd $script_dir/RamananCodes
%demoneimagewhole_scaled($path1, $path2)

'..........Registeration..........'
mkdir $data_root_dir/Registeration
mkdir $data_root_dir/Registeration/$1
cd $script_dir/RegisterationCodes
FindAveragePointsDK($path2, $path3)
mapTFD2ICML($path1, $path3)

'..........Preprocessing..........'
mkdir $data_root_dir/IS
mkdir $data_root_dir/IS/$1
cd $script_dir/IsotropicCodes/INface_tool
install_INface
cd ../
preprocessing_IS($path3, $path4)
exit
EOF
#end of Matlab commands 1


#'..........Img2Batch..........'
pwd
mkdir $data_root_dir/batches
mkdir $data_root_dir/batches/$1
cd $model_dir/working-cuda-convnet-2013-11-15/data
pwd
$path5
$path6
python $script_dir/Dimg2batch.py $path5 $path6
# TODO: put the right mean face.
# This one is the mean of the initial test set.
cp $model_dir/working-cuda-convnet-2013-11-15/data/AFEWIS/batch48/batches.meta $path6

#'..........ConvNet..........'
cd ../
python pif_export.py -f ./data/tmp/ConvNet__2013-07-17_13.27.15 --export=afew --show-preds=probs --test-range=1-1 --test-data-path=$path6

# svm EXPERIMENT 4
'..........SVM..........'
matlab -nodesktop << EOF       #starts Matlab
warning off
mkdir $data_root_dir/SVM
mkdir $data_root_dir/SVM/$1
cd $script_dir/SVMCodes
addpath(genpath('./libsvm-3.17'))
%path
%This part does not work in parallel (overwritting convnet output)
useSVM4($path7)
exit
EOF

#end of Matlab commands 2
