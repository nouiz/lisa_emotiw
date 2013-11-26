#!/usr/bin/env python 

import sys
import os
import subprocess
from string import Template



try:
    NO_JOBS, DATA_PATH, CLIP_ID  = sys.argv[1:]
    NO_JOBS = int(NO_JOBS)
except:
    print "Usage %s no_jobs data_path clip_id" % sys.argv[0]
    sys.exit(1)

BASE_PATH='/usagers/bornj/lisa_emotiw/scripts/jorg/remote'

job_tmpl = Template(
"""#!/bin/sh
#PBS -l nodes=1,walltime=1:00:00,mem=3gb
#PBS -j oe

cd $BASE_PATH/RamananCodes
matlab -nodesktop << EOF


imgpath='$DATA_PATH/$CLIP_ID/*.png'

imgdir=dir(imgpath);
disp(['Number of frames: ' length(imgdir)])
while length(imgdir)>0
    rng('shuffle')
    k = randi(length(imgdir))

    disp(['$DATA_PATH/$CLIP_ID/' imgdir(k).name])
    demoneimagewhole_scaled(['$DATA_PATH/$CLIP_ID/' imgdir(k).name])

    imgdir=dir(imgpath)
end

disp('$DATA_PATH/$CLIP_ID/DONE.txt')
fid = fopen('$DATA_PATH/$CLIP_ID/DONE.txt', 'w')
fclose(fid)
EOF

""")


for j in xrange(NO_JOBS):
    # Generate Job-Script
    subs =  {
        'BASE_PATH': BASE_PATH,
        'DATA_PATH': '~/'+DATA_PATH,
        'CLIP_ID':  CLIP_ID,
        'WORK_ID':  j,
    }

    job_script = job_tmpl.substitute(subs)

    p = subprocess.Popen(['qsub', "--"], stdin=subprocess.PIPE)
    p.stdin.write(job_script)
    p.communicate()
    p.stdin.close()

