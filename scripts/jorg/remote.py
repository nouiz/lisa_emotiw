#!/usr/bin/env python

import os
import shutil
import subprocess
import sys
from time import sleep

REMOTE_USER_HOST='bornj@cudahead.rdgi.polymtl.ca'
REMOTE_DATA_PATH='tmp/jobdata/'                               # directory there the workpackages will be copied
REMOTE_JOB_PATH='lisa_emotiw/scripts/jorg/remote/ramanan.job' # jobscript to submit on the cluster

REMOTE_PARALEL_JOBS=1
REMOTE_POLLING_TIMEOUT=10


#=============================================================================

def submit_workpackage(local_dir):
    # ensure target directory exists
    run_remote(['mkdir', '-p', REMOTE_DATA_PATH])

    # copy workpackage
    print "Copy workpackage to cluster... [%s:%s]" % (REMOTE_USER_HOST, REMOTE_DATA_PATH)
    remote_dir = REMOTE_DATA_PATH+local_dir
    rsync_local_remote(local_dir, remote_dir )

    # enqueue remote work on cluster 
    print "Submitting %d jobs to queuing system on cluster..." % REMOTE_PARALEL_JOBS
    qsub_cmd = ['qsub', REMOTE_JOB_PATH]
    for j in xrange(REMOTE_PARALEL_JOBS):
        run_remote(qsub_cmd)

    # wait for workpackage to be completed
    print "Waiting for workpackage to be completed..."
    test_cmd = ['test', '-e', REMOTE_DATA_PATH+local_dir+'/DONE' ]
    while run_remote(test_cmd, except_on_error=False) == 1:
        sleep(REMOTE_POLLING_TIMEOUT)

    # copy results back to local machine
    print "Copy results back to local machine.,."


#=============================================================================
# Remote execution and copying 
class RemoteExecutionError(RuntimeError):
    def __init__(self, cmd, retcode):
        RuntimeError.__init__('Execution of "%s" returned %d' % (' '.join(cmd), retcode))

def run_remote(cmd, except_on_error=True):
    """ Use ssh to run the specified cmd on REMOTE_USER_HOST.
        :cmd: has to be either a list or a string.
    """
    if isinstance(cmd, str):
        cmd = [cmd]
    cmd = ['ssh', REMOTE_USER_HOST] + cmd
    retcode = subprocess.call(cmd)

    if except_on_error and retcode != 0:
        raise RemoteExecutionError(cmd, retcode)
    return retcode

def rsync_local_remote(local_path, remote_path):
    remote = "%s:%s" % (REMOTE_USER_HOST, remote_path)
    cmd = ['rsync', '-r', local_path, remote]
    print cmd
    retcode = subprocess.call(cmd)
    if retcode != 0:
        raise RemoteExecutionError(cmd, retcode)

def rsync_remote_local(remote_path, local_path):
    remote = "%s:%s" % (REMOTE_USER_HOST, remote_path)
    cmd = ['rsync', '-r', remote, local_path]
    retcode = subprocess.call(cmd)
    if retcode != 0:
        raise RemoteExecutionError(cmd, retcode)

if __name__  == "__main__":
    submit_workpackage('test.lala')
