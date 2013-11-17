'''
Created on 2013-06-17

@author: JP Raymond (raymonjp@iro)

Original version that uses watchdog, which seems to fail to observe a shared
folder on a Windows virtual mahine.
'''

import time
import logging
import os
import shutil

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

WORKING_DIRECTORY = 'Z:/sans-bkp/faces/picasa_process'
PROCESSING_DIRECTORY = os.path.join(WORKING_DIRECTORY,
                                    'ProcessingFolder (Do Not Touch)')
    # Folder used for processing (the only folder sync'ed with Picasa).
TO_PROCESS_EXT = 'process_me' # Extension of a folder that needs to be processed.
PROCESSED_EXT = 'processed' # Extension of a folder that has been processed.
FACES_EXT = 'faces' # Extension of the folder where the faces are saved.
FACES_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'Faces')
    # Folder where the images extracted by Picasa will be saved.
PICASA_FILE_TYPES = frozenset(['jpeg', 'jpg', 'bmp', 'gif', 'png', 'tga',
                               'tif', 'tiff', 'webp']) # Really necessary? ...
AHK_EXE = 'Picasa_ahk.exe'
INIT_AHK_EXE = 'PicasaInit_ahk.exe'

class DirToProcessEventHandler(FileSystemEventHandler):
    
    '''
    # It's been decided that only the renaming of a folder should trigger the
    # process.
    def on_created(self, event):
        super(DirToProcessEventHandler, self).on_created(event)
        if event.is_directory:
            self.__process(event.src_path)
    '''
        
    def on_moved(self, event):
        super(DirToProcessEventHandler, self).on_moved(event)
        if os.path.isdir(event.dest_path):
            self.__process(event.dest_path)
        
    # Handling cases where user picks folder names that will lead to duplicates.
    #   - 'SomeFolderName.process_me' when 'SomeFolderName.processed' (or.faces)
    #     already exists ...
    #   - 'Faces' ...
    #   - 'ProcessingFolder (Do Not Touch)' ...
    def __process(self, path):        
        path_w_out_folder_ext, folder_ext = os.path.splitext(path)
        if folder_ext[1:] == TO_PROCESS_EXT:
            os.system(INIT_AHK_EXE)
	    os.rename(path, PROCESSING_DIRECTORY)
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
            os.rename(PROCESSING_DIRECTORY, path_w_out_folder_ext + '.' + PROCESSED_EXT)
	    time.sleep(2)	# For same purpose as above, in case a 2nd run of the process occurs immediately
				# after this one.

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    event_handler = DirToProcessEventHandler()
    observer = Observer()
    observer.schedule(event_handler, WORKING_DIRECTORY, False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
