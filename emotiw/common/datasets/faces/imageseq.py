

class ImageSequenceDataset(object):
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
    
