'''
Created on 2013-06-17

@author: JP Raymond (raymonjp@iro)

Adapted from the original version so that it doesn't require the watchdog
library.
'''

import time
import logging
import os
import shutil

WORKING_DIRECTORY = 'E:/PicasaProcess'
LOCAL_WORKING_DIRECTORY = 'C:/PicasaProcess Scripts'
PROCESSING_DIRECTORY = os.path.join(LOCAL_WORKING_DIRECTORY,
                                    'ProcessingFolder (Do Not Touch)')
    # Folder used for processing (the only folder sync'ed with Picasa).
TO_PROCESS_EXT = 'process_me' # Extension of a folder that needs to be processed.
IN_PROCESS_EXT = 'in_process' # Extension of a folder that is being processed.
PROCESSED_EXT = 'processed' # Extension of a folder that has been processed.
FACES_EXT = 'faces' # Extension of the folder where the faces are saved.
FACES_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Faces')
    # Folder where the images extracted by Picasa will be saved.
PICASA_FILE_TYPES = frozenset(['jpeg', 'jpg', 'bmp', 'gif', 'png', 'tga',
                               'tif', 'tiff', 'webp']) # Really necessary? ...
AHK_EXE = 'Picasa_ahk.exe'
INIT_AHK_EXE = 'PicasaInit_ahk.exe'

# Handling cases where user picks folder names that will lead to duplicates.
#   - 'SomeFolderName.process_me' when 'SomeFolderName.processed' (or.faces)
#     already exists ...
#   - 'Faces' ...
def __process(path):
    os.system(INIT_AHK_EXE) # Don't really feel like explaining the purpose of
                            # that :s ...
    path_w_out_folder_ext, folder_ext = os.path.splitext(path)
    in_process_folder_path = path_w_out_folder_ext + '.' + IN_PROCESS_EXT
    os.rename(path, in_process_folder_path)
    shutil.copytree(in_process_folder_path, PROCESSING_DIRECTORY)      
    file_count = 0
    for f in os.listdir(PROCESSING_DIRECTORY):
        file_count += 1
        name, ext = os.path.splitext(f)
        if ext[1:] in PICASA_FILE_TYPES:
            # We add '_picassa' in case Picasa appends digits to the file names.
            f_ = name + '_picassa' + ext 
            os.rename(os.path.join(PROCESSING_DIRECTORY, f),
                      os.path.join(PROCESSING_DIRECTORY, f_))
    time.sleep(2)   # It seems like some amount of time must pass between two
                    # runs of Picasa. The first run occurs in INIT_AHK_EXE.
    os.system(AHK_EXE + ' ' + str(file_count))
    os.rename(FACES_DIRECTORY, path_w_out_folder_ext + '.' + FACES_EXT)
    os.rename(in_process_folder_path,
              path_w_out_folder_ext + '.' + PROCESSED_EXT)
    shutil.rmtree(PROCESSING_DIRECTORY)
    time.sleep(2)   # To make sure PROCESSING_DIRECTORY will be deleted on an
                    # immediate subsequent process. Could be cleaner I guess ...
                    
if __name__ == '__main__':
    while True:
        for d in os.walk(WORKING_DIRECTORY).next()[1]:
            if d.endswith('.' + TO_PROCESS_EXT):
                __process(os.path.join(WORKING_DIRECTORY, d))
        time.sleep(1)
