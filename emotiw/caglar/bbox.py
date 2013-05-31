from tables import *

"""
Bounding boxes class.
"""
class BoundingBox(IsDescription):
    picasaBatchNumber = Int32Col()
    idxInPicasaBatch = Int32Col()
    faceno = Int32Col()
    imgno = Int32Col()
    row = Int32Col()
    col = Int32Col()
    height = Int32Col()
    width = Int32Col()

