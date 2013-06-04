import numpy
import os

from output_map import OutputMapFile, OutputMap, CroppedPatchesDataset

def create_dataset():
    """
    Create the dataset file.
    """
    path = "/data/lisatmp/data/faces_bbox/"
    out_file = "test_outputmap.h5"
    outputmap_file = OutputMapFile(path=path)
    n_examples=3000
    out_shape=(3000, 100, 100)
    h5file, gcols = OutputMapFile.create_file(path=path, filename=out_file, n_examples=n_examples, out_shape=out_shape)

    pt = numpy.random.randint(255, size=out_shape)
    ino = numpy.random.rand(n_examples)
    iloc = numpy.random.rand(n_examples)
    tgt = numpy.random.rand(n_examples)

    outputmap_file.save_output_map(h5file, pt , ino, iloc, tgt)
    h5file.close()

def access_dataset(out_shape=(3000, 100, 100)):
    path = "/data/lisatmp/data/faces_bbox/"
    out_file = "test_outputmap.h5"
    filename = os.path.join(path, out_file)
    print filename

    croppedPatchesDataset = CroppedPatchesDataset(img_shape=out_shape, h5_file=filename)
    import ipdb; ipdb.set_trace()

if __name__=="__main__":
    access_dataset()
