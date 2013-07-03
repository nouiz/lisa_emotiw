# Copyright (c) 2013 University of Montreal, Pascal Vincent,
# Vincent Archambault, Pascal Lamblin
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of the authors and contributors to this software may not be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

class ImageSequenceDataset(object):

    def __init__(self, name):
        self.name = name

    def get_name(self):        
        return self.name

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        raise NotImplementedError(str(type(self)) + " does not implement __len__.")

    def __getitem__(self, i):
        """
        Returns the ith sequence. Can also be called with a range or a list, to get subsets of this dataset.
        """

        if isinstance(i, int):
            if i < 0 or i >= self.__len__():
                raise IndexError()
            return self.get_sequence(i)

        elif isinstance(i, slice):
            return ImageSequenceSubset(self,range(i.start, i.stop, i.step))

        else: # assume list of indexes
            li = [idx for idx in i] # build list fro any iterable
            return ImageSequenceSubset(self,li)

    def get_sequence(self, i):
        """Returns a sequence of image frames that will typically be a subclass of FaceImagesDataset.
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
        print "( None: %6d / %d, \t %.2f%% )" % ( none_count, length, 100.0*none_count/length)
        not_none_count = length-none_count
        for val in values_counts:
            print >>out, "%30s: %6d / %d \t (%.2f%%)" % (val, values_counts[val], not_none_count, 100.0*values_counts[val]/not_none_count)
        print >>out


# Helper classes

class ImageSequenceSubset(ImageSequenceDataset):
    """
    A subset view of an ImageSequenceDataset. This view is itself a ImageSequenceDataset.
    """
    def __init__(self, orig_dataset, indices, name=None):
        if name is None:
            name = "subset of "+orig_dataset.name
        super(ImageSequenceSubset,self).__init__(name)
        self.orig_dataset = orig_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
        
    def get_sequence(self, i):
        return self.orig_dataset.get_sequence(self.indices[i])        

    def get_label(self, i):
        return self.orig_dataset.get_label(self.indices[i])        
        


class SimpleImageSequenceDataset(ImageSequenceDataset):

    def __init__(self, name, image_sequence_list, label_list=None):
        super(SimpleImageSequenceDataset,self).__init__(name)
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
