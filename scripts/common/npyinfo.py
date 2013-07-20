#!/usr/bin/env python

import sys
import numpy as np

#!/usr/bin/env python
def npyinfo(fname):
    m = np.load(fname)

    print "----------------"
    print m
    print "----------------"
    print "  filename:", fname
    print "  shape:", m.shape
    print "  dtype:", m.dtype
    print "----------------"

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print "Provide a single .npy file as argument"
    else:
        npyinfo(sys.argv[1])
