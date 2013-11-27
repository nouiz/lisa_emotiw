#!/usr/bin/env python 

import sys
import os
import subprocess
from string import Template



try:
    N_JOBS, DATA_PATH, CLIP_ID  = sys.argv[1:]
    N_JOBS = int(N_JOBS)
except:
    print "Usage %s no_jobs data_path clip_id" % sys.argv[0]
    sys.exit(1)

BASE_PATH='/usagers/bornj/lisa_emotiw/scripts/jorg/remote'

job_tmpl = Template(
"""#!/bin/sh
#PBS -l nodes=1,walltime=1:00:00,mem=2500mb
#PBS -j oe

cd $BASE_PATH/RamananCodes
matlab -nodesktop << EOF

imgpath='$DATA_PATH/$CLIP_ID/*.png';
n_jobs=$N_JOBS;
job_id=$JOB_ID+1;

imgdir=dir(imgpath);
n_frames=length(imgdir);
disp(['Number of frames: ' num2str(n_frames)])

disp('Deterministically pick frames');
for k=job_id:n_jobs:n_frames
    img=imgdir(k).name;

    disp(['Processing frame ' img]);
    demoneimagewhole_scaled(['$DATA_PATH/$CLIP_ID/' img]);
end 

disp('Randomly pick frames from the directory');
imgdir=dir(imgpath);
while length(imgdir)>0
    disp(['No. of frames left to pick from: ' num2str(length(imgdir))])
    rng('shuffle');
    k = randi(length(imgdir));

    disp(['$DATA_PATH/$CLIP_ID/' imgdir(k).name]);
    demoneimagewhole_scaled(['$DATA_PATH/$CLIP_ID/' imgdir(k).name]);

    imgdir=dir(imgpath);
end

if length(imgdir) == 0
    disp('$DATA_PATH/$CLIP_ID/DONE.txt');
    fid = fopen('$DATA_PATH/$CLIP_ID/DONE.txt', 'w');
    fclose(fid);
end

disp('I was kind of a zombieee ... whee');
EOF

""")


for j in xrange(N_JOBS):
    # Generate Job-Script
    subs =  {
        'BASE_PATH': BASE_PATH,
        'DATA_PATH': '~/'+DATA_PATH,
        'CLIP_ID':   CLIP_ID,
        'N_JOBS':    N_JOBS,
        'JOB_ID':    j,
    }

    job_script = job_tmpl.substitute(subs)

    p = subprocess.Popen(['qsub', "--"], stdin=subprocess.PIPE)
    p.stdin.write(job_script)
    p.communicate()
    p.stdin.close()

