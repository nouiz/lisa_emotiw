% This is the install script for the INface toolbox v2.0. 
% 
% All this script does is 
% adding the toolbox paths to Matlabs search path and compiles the C code to 
% produce the mex files needed by the toolbox. It should be noted that all 
% components of the toolbox were tested with Matlab version 7.5.0.342 
% (R2007b) running on WindowsXP Professional (SP3) OS and later with Matlab 
% version 7.11.0.584 (R2010b) running on Windows 7.
% 
% I have not tested any of the code on a Linux machine. Nevertheless, I see 
% no reason why it should not work on Linux as well (except maybe for the 
% part were the path is added in this script). If this script fails 
% just compile the c/c++ code yourself and add the path to the toolbox together 
% with all subdirectories to Matlabs search path.
% 
% As always, if you haven't done so yet, type "mex -setup" prior to running 
% this script to select an appropriate compiler from the ones available to 
% you.
% 
% In case compiling of the c/c++ code fails, the toolbox still works
% normally. The only things affected are the non-local means and adaptive
% non-local means techniques which require the compiled mex files.


%% Get current directory and add all subdirectories to path
current = pwd;
addpath(current);
addpath([current '/auxilary']);
addpath([current '/mex']);
addpath([current '/histograms']);
addpath([current '/photometric']);
addpath([current '/postprocessors']);
addpath([current '/demos']);
savepath


%% Produce the mex (matlab executables) files
mex mex/perform_nlmeans_mex.cpp
mex mex/perform_nlmeans_mex1.cpp