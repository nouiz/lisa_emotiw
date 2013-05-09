import sys

class ImageSequenceDataset(object):

    def get_name(self):
        return ""

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        raise NotImplementedError(str(type(self)) + " does not implement __len__.")

    def __getitem__(self, i):
        """
        Returns an object that exposes all properties of the ith face example
        Technically it will return an instance of FaceDatasetExample on which
        accessing properties will be handled by calling the corresponding get_... method
        of the current dataset, with parameter i.
        """

        if isinstance(i, int):
            if i < 0 or i >= self.__len__():
                raise IndexError()
            return self.get_sequence(i)

        elif isinstance(i, slice):
            raise NotImplementedError(str(type(self)) + " does not implement __getitem__ for a slice")
            # return ImageSequenceSubset(self,range(i.start, i.stop, i.step))

        else: # assume list of indexes
            raise NotImplementedError(str(type(self)) + " does not implement __getitem__ for a list of indexes")
            # li = [idx for idx in i] # build list fro any iterable
            # return FaceSequenceSubset(self,li)

    def get_sequence(self, i):
        """Returns a sequence of image frames that will typically be a subclass of FaceImagesDataset.
        Warning: if a given image form a video contains several detected faces (as per bbox) it may be repeated several consecutive times
        with the associated bounding box info changing.
        """
        raise NotImplementedError(
            "get_sequence not yet implemented in class " + str(type(self)) + " please implement it")

    def get_label(self, i):
        """returns the label for the whole sequence, as a string"""
        pass

    def get_standard_train_test_splits(self):
        """Returns a list of pairs (train_indexes, test_indexes) where train_indexes and test_indexes are themselves lists or integer ndarrays
        (containing indexes of sequence that can be passed to get_sequence)
        Returns None if not available"""
        return None

    def count_values(self, method):
        """Returns a dictionary mapping values returned by a method call to their count over the __len__ examples
        Ex:
        dataset.count_values(dataset.get_label)
        """
        counts = {}
        for i in xrange(self.__len__()):
            try:
                feature = method(i)
            except:
                pass
            if feature not in counts:
                counts[feature] = 1
            else:
                counts[feature] += 1
        return counts
    
    def print_info(self, out=sys.stdout):
        """Prints various info and statistics about this dataset, such as class counts"""

        length = self.__len__()
        print >>out, "**********************************************"
        print >>out, "IMAGE SEQUENCE DATASET ", self.get_name()
        print >>out, "length (# examples):", length

        # report split counts
        splits = self.get_standard_train_test_splits()
        splitcounts = None
        if splits is not None:
            splitcounts = []
            for split in splits:
                splitcounts.append( [ len(indices) for indices in split ] )
        print >>out, "standard splits:"
        print >>out, splitcounts
        print >>out

        print "Label counts:"
        values_counts = self.count_values(self.get_label)
        none_count  = 0
        if None in values_counts:
            none_count = values_counts[None]
            del values_counts[None]
        print "( None:", none_count, "/", length, ")"
        not_none_count = length-none_count
        for val in values_counts:
            print >>out, "%30s: %d \t (%.2f%%)" % (val, values_counts[val], 100.0*values_counts[val]/not_none_count)
        print >>out

        


class SimpleImageSequenceDataset(ImageSequenceDataset):

    def __init__(self, name, image_sequence_list, label_list=None):
        self.name = name
        self.image_sequence_list = image_sequence_list
        self.label_list = label_list

    def get_name(self):
        return self.name

    def __len__(self):
        return len(self.image_sequence_list)

    def get_sequence(self, i):
        return self.image_sequence_list[i]

    def get_label(self, i):
        if self.label_list is None:
            return None
        return self.label_list[i]
