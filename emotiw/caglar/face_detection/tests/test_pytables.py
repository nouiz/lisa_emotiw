import tables


h5file = tables.openFile("tutorial1.h5", mode = "w", title = "Test file")
group = h5file.createGroup("/", 'detector', 'Detector information')
ca = h5file.createCArray(h5file.root, "carray", tables.UInt8Atom(), shape=(2048, 8, 8))


