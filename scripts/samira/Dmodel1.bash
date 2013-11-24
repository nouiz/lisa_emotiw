# Ramanan step
# inputs = facetubes_path, ramanan_path, avg_pts_path, preprocessed_path
# testvideo/ Ramanan/testvideo/ avgpoint.mat


path1="'../$1'"
path2="'../Ramanan/$1'"
path3="'../Registeration/$1'"
path4="'../IS/$1'"
path5="../../IS/$1"
path6="../../batches/$1"
path7="'../SVM/$1'"


matlab -nodesktop << EOF       #starts Matlab
warning off
mkdir Ramanan
cd ./Ramanan
mkdir $1
cd ../RamananCodes
demoneimagewhole_scaled($path1, $path2)

'..........Registeration..........'
cd ../
mkdir Registeration
cd ./Registeration
mkdir $1
cd ../RegisterationCodes
FindAveragePointsDK($path2, $path3)
mapTFD2ICML($path1,$path3)

'..........Preprocessing..........'
cd ../
mkdir IS
cd ./IS
mkdir $1
cd ../IsotropicCodes/INface_tool
install_INface
cd ../
preprocessing_IS($path3,$path4)
exit
EOF
#end of Matlab commands 1


#'..........Img2Batch..........'
pwd
mkdir ./batches
mkdir ./batches/testvideo
cd ./working-cuda-convnet-2013-11-15/data
pwd
$path5 
$path6
python Dimg2batch.py $path5 $path6


#'..........ConvNet..........'
cd ../
#python pif_export.py -f ./data/tmp/ConvNet__2013-07-17_13.27.15 --export=afew --show-preds=probs --test-range=1-8 --test-data-path=./data/AFEWIS/batch48 --multiview-test=0 --logreg-name=logprob


# svm EXPERIMENT 4
'..........SVM..........'
matlab -nodesktop << EOF       #starts Matlab
warning off
cd ../
mkdir SVM
cd ./SVM
mkdir $1
cd ../SVMCodes
addpath(genpath('./libsvm-3.17'))
%path
%This part does not work in parallel (overwritting convnet output)
useSVM4($path7)

exit
EOF                              

#end of Matlab commands 2
