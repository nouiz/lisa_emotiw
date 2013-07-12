import emotiw.raymonjp.ramanan as ramanan

ramanan.process_first_file('/some/folder/containing/images/only/',
                           '/some/folder/where/to/save/the/results/',
                           '/some/folder/where/to/move/the/processed/file/',
                           1)	# Model number (1, 2, or 3 - see ramanan.py for
                                # more info). If not specified, model number 3,
                                # the most performant but very slow, will be
                                # used.
