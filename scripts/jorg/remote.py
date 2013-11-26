#!/usr/bin/env python

import logging
import subprocess
from time import sleep

#=============================================================================
# Remote execution and copying 
class RemoteExecutionError(RuntimeError):
    def __init__(self, cmd, retcode):
        RuntimeError.__init__(self, 'Execution of "%s" returned %d' % (' '.join(cmd), retcode))

def run_remote(cmd, except_on_error=True):
    """ Use ssh to run the specified cmd on REMOTE_USER_HOST.
        :cmd: has to be either a list or a string.
    """
    if isinstance(cmd, str):
        cmd = [cmd]
    cmd = ['ssh', REMOTE_USER_HOST] + cmd
    logging.debug("Calling %s", cmd)
    retcode = subprocess.call(cmd)

    if except_on_error and retcode != 0:
        raise RemoteExecutionError(cmd, retcode)
    return retcode

def rsync_local_remote(local_path, remote_path):
    remote = "%s:%s" % (REMOTE_USER_HOST, remote_path)
    cmd = ['rsync', '-r', local_path, remote]
    logging.debug("Calling %s", cmd)
    retcode = subprocess.call(cmd)
    if retcode != 0:
        raise RemoteExecutionError(cmd, retcode)

def rsync_remote_local(remote_path, local_path):
    remote = "%s:%s" % (REMOTE_USER_HOST, remote_path)
    cmd = ['rsync', '-r', remote, local_path]
    logging.debug("Calling %s", cmd)
    retcode = subprocess.call(cmd)
    if retcode != 0:
        raise RemoteExecutionError(cmd, retcode)

