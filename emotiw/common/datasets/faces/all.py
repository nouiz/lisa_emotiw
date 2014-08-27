from emotiw.common.datasets.faces.tfd import datasets_constructors_list as tfd_datasets_constructors_list

from BioID import BioID
from multipie import MultiPie
from aflw import AFLW 
from afw import AFW as AFW_v1
from afw_v2 import AFW as AFW_v2
from caltech import Caltech
from lfpw import Lfpw
from FDDB import FDDB
from googleFaceDataset import GoogleFaceDataset
from googleEmotionDataset import GoogleEmotionDataset

from HIIT6HeadPose import HIIT6HeadPose
from IHDPHeadPose import IHDPHeadPose
from inrialpesWrapper import InrialpesHeadPose

from concatDatasets import ConcatDatasets

from emotiw.common.datasets.faces.afew import AFEWImageSequenceDataset
from emotiw.common.datasets.faces.afew2 import AFEW2ImageSequenceDataset
from emotiw.common.datasets.faces.afew2_test import AFEW2TestImageSequenceDataset
from emotiw.common.datasets.faces.new_clips import NewClipsImageSequenceDataset

# The following are lists of triples (dataset_name, dataset_constructor, description)

def concatDatasets():
    return ConcatDatasets([BioID, GoogleEmotionDataset, AFEWImageSequenceDataset])

image_datasets_constructors_list = [ ("AFLW", AFLW, ""),
                                     ("AFW_v1", AFW_v1, ""),
                                     ("AFW_v2", AFW_v2, ""),
                                     ("Caltech", Caltech, ""),
                                     ("ConcatDatasets", concatDatasets, ""),
                                     ("MultiPie", MultiPie, ""),
                                     ("BoiID", BioID, ""),
                                     ("FDDB", FDDB, ""),
                                     ("GoogleFaceDataset", GoogleFaceDataset, ""),
                                     ("GoogleEmotionDataset", GoogleEmotionDataset, ""),
                                     ("HIIT6HeadPose", HIIT6HeadPose, ""),
                                     ("IHDPHeadPose",IHDPHeadPose, ""),
                                     ("InrialpesHeadPose",InrialpesHeadPose, ""),
                                     ("LFPW", Lfpw, "")
                                     ] + tfd_datasets_constructors_list

image_sequence_datasets_constructors_list = [
    ("AFEWImageSequenceDataset", AFEWImageSequenceDataset, ""),
    ("AFEW2ImageSequenceDataset", AFEW2ImageSequenceDataset, ""),
    ("AFEW2TestImageSequenceDataset", AFEW2TestImageSequenceDataset, ""),
    ("NewClipsImageSequenceDataset", NewClipsImageSequenceDataset, "")]

instantiated_datasets = {}

def select_dataset(dataset_constructors_list,
                   catch_and_try_again=True):
    """Lets the user select a dataset from the given list.
    The call will try to instantiate that dataset (if not already instantiated)
    and return it. It remembers the already instantiated datasets by populating
    and looking them up in the instantiated_datasets dictionary.
    If catch_and_try_again is true, the call will not raise an exception in case of a problem,
    during instantiation, but will instead re-ask the user for a dataset until it can successfully
    return it."""

    while True:
        print
        print "****************************"
        print "***  Available datasets  ***"
        print "****************************"
        print
        for i,triple in enumerate(dataset_constructors_list):
            name, constructor, descr = triple
            print i,':',name, "  \t",descr
        print
        print "Choose one to instantiate: ",
        try:
            idx = int(raw_input())
            name, constructor, descr = dataset_constructors_list[idx]
            if name in instantiated_datasets:
                dataset =  instantiated_datasets[name]
            else:
                print "Instantiating ",name, "..."
                dataset = constructor()
                instantiated_datasets[name] = dataset
            print "OK."
            print
            return dataset

        except Exception, e:
            if catch_and_try_again:
                print "  Failed."
                print e
            else:
                raise
            
        
