#!/bin/bash

path_in="'$1'"
path_out="'$2'"
script_dir="$3"
current_dir="$4"

cd $script_dir

matlab -nodesktop << EOF       #starts Matlab
warning off

$path_in, $path_out
demoneimagewhole_alt($path_in, $path_out)

exit
EOF

cd $current_dir
