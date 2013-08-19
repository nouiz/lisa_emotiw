import sys
from pylearn2.utils import serial

def go(fname):
   train = serial.load(fname)
   train.main_loop()
   train.save()

if __name__=='__main__':
    go(sys.argv[1])
