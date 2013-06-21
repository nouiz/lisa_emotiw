import PIL.Image
from crop_face import crop_face
from faceimages import FaceDatasetExample, keypoints_names
import numpy
import tables

def hdf5_as_npy(files, save_as):
    #save_as: save file prefix (save_as = '/hello/X' -> generate /hello/X_train_x.npy, /hello/X_train_y.npy, /hello/X_test_x.npy, /hello/X_test_y.npy)
    FLUSH_DELAY_IN_LINES = 25000
    
    test_sizes = []
    train_sizes = []
    the_files = []
    rng = numpy.random.RandomState()
    
    for f in files:
        the_files.append(tables.openFile(f))
        the_file = the_files[-1]

        train_sizes.append(len(the_file.root.train.img))
        if the_file.root.test._v_nchildren != 0:
            test_sizes.append(len(the_file.root.test.img))
        else:
            test_sizes.append(0)

    print 'preparing to dump', sum(train_sizes), 'training samples and', sum(test_sizes), 'testing samples...'

        
    #May only memmap one file at a time
    out_train_x = numpy.memmap(save_as + '_train_x.npy', mode='write', shape=(sum(train_sizes), 96*96*3))

    idx_array = range(sum(train_sizes))

    flush_delay_in_lines = FLUSH_DELAY_IN_LINES

    print 'beginning dump of train data.'

    for i in xrange(sum(train_sizes)):
        if flush_delay_in_lines <= 0:
            flush_delay_in_lines = FLUSH_DELAY_IN_LINES
            #out_train_x.flush()
        idx_idx = rng.randint(sum(train_sizes) - i)
        idx = idx_array[i+idx_idx]
        idx_array.remove(idx)
        idx_array.insert(0, idx)
        #Select a random idx within the array of possible
        #idx, within the available idx (that is, among [i, len(idx_array)]).
        #The value at this location is a new, valid idx within the train data
        #by construction. This avoids the pitfall of usual random selection of
        #elements where, when i is large, the expectation of the number of trials
        #before we get an index that wasn't taken before is exponential w.r.t. i.
    
        the_file = None
        t_size = 0
        for f_idx, x in enumerate(train_sizes):
            if t_size + x > idx:
                the_file = the_files[f_idx]
                break
            else:
                t_size += x

        char_arr_image = [len(x) != 0 and ord(x) or 0 for x in the_file.root.train.img[idx-t_size]]

        unflipped_image = []
        for idx, _ in enumerate(char_arr_image[::3]):
            unflipped_image.append(char_arr_image[3*idx+2])
            unflipped_image.append(char_arr_image[3*idx+1])
            unflipped_image.append(char_arr_image[3*idx])

        for l_num, _ in enumerate(unflipped_image[::96*3]):
            for idx in xrange(96/2):
                temp = unflipped_image[l_num*96*3 + idx*3:l_num*96*3 + idx*3 + 3]
                unflipped_image[l_num*96*3 + idx*3:l_num*96*3 + idx*3 + 3] = unflipped_image[l_num*96*3 + 96*3 - idx*3 - 3:l_num*96*3 + 96*3 - idx*3]
                unflipped_image[l_num*96*3 + 96*3 - idx*3 - 3:l_num*96*3 + 96*3 - idx*3] = temp

        unflipped_image.reverse()

        out_train_x[i, :] = unflipped_image       
        flush_delay_in_lines -= 1
        #NOTE: numpy converts '\0' to '' because it encodes strings as cstrings.
    
    del out_train_x

    flush_delay_in_lines = FLUSH_DELAY_IN_LINES
    out_train_y = numpy.memmap(save_as + '_train_y.npy', dtype=numpy.float32, mode='write', shape=(sum(train_sizes), len(keypoints_names), 2))
    for i, idx in enumerate(idx_array):
        if flush_delay_in_lines <= 0:
            flush_delay_in_lines = FLUSH_DELAY_IN_LINES
            out_train_y.flush()
        the_file = None
        t_size = 0
        for f_idx, x in enumerate(train_sizes):
            if t_size + x > idx:
                the_file = the_files[f_idx]
                break
            else:
                t_size += x

        out_train_y[i, :] = numpy.asarray(the_file.root.train.label[idx-t_size])
        flush_delay_in_lines -= 1

    del out_train_y
    print 'train data dumped successfully.'
    if sum(test_sizes) == 0:
        return
    out_test_x = numpy.memmap(save_as + '_test_x.npy', mode='write', shape=(sum(test_sizes), 96*96*3))

    idx_array = range(sum(test_sizes))

    flush_delay_in_lines = FLUSH_DELAY_IN_LINES
    
    print 'beginning dump of test data.'
    for i in xrange(sum(test_sizes)):
        if flush_delay_in_lines <= 0:
            flush_delay_in_lines = FLUSH_DELAY_IN_LINES
            out_test_x.flush()

        idx_idx = rng.randint(sum(test_sizes) - i)
        idx = idx_array[i+idx_idx]
        idx_array.remove(idx)
        idx_array.insert(0, idx)
        #Select a random idx within the array of possible
        #idx, within the available idx (that is, among [i, len(idx_array)]).
        #The value at this location is a new, valid idx within the test data
        #by construction. This avoids the pitfall of usual random selection of
        #elements where, when i is large, the expectation of the number of trials
        #before we get an index that wasn't taken before is exponential w.r.t. i.

        the_file = None
        t_size = 0
        for f_idx, x in enumerate(test_sizes):
            if t_size + x > idx:
                the_file = the_files[f_idx]
                break
            else:
                t_size += x


        char_arr_image = [len(x) != 0 and ord(x) or 0 for x in the_file.root.test.img[idx-t_size]]

        unflipped_image = []
        for idx, _ in enumerate(char_arr_image[::3]):
            unflipped_image.append(char_arr_image[3*idx+2])
            unflipped_image.append(char_arr_image[3*idx+1])
            unflipped_image.append(char_arr_image[3*idx])

        for l_num, _ in enumerate(unflipped_image[::96*3]):
            for idx in xrange(96/2):
                temp = unflipped_image[l_num*96*3 + idx*3:l_num*96*3 + idx*3 + 3]
                unflipped_image[l_num*96*3 + idx*3:l_num*96*3 + idx*3 + 3] = unflipped_image[l_num*96*3 + 96*3 - idx*3 - 3:l_num*96*3 + 96*3 - idx*3]
                unflipped_image[l_num*96*3 + 96*3 - idx*3 - 3:l_num*96*3 + 96*3 - idx*3] = temp

        unflipped_image.reverse()

        out_test_x[i, :] = unflipped_image 
        flush_delay_in_lines -= 1

    del out_test_x

    flush_delay_in_lines = FLUSH_DELAY_IN_LINES
    out_test_y = numpy.memmap(save_as + '_test_y.npy', dtype=numpy.float32, mode='write', shape=(sum(test_sizes), len(keypoints_names), 2))
    for i, idx in enumerate(idx_array):
        if flush_delay_in_lines <= 0:
            flush_delay_in_lines = FLUSH_DELAY_IN_LINES
            out_test_y.flush()

        the_file = None
        t_size = 0
        for f_idx, x in enumerate(test_sizes):
            if t_size + x > idx:
                the_file = the_files[f_idx]
                break
            else:
                t_size += x

        out_test_y[i, :] = numpy.asarray(the_file.root.test.label[idx-t_size])
        flush_delay_in_lines -= 1
    del out_test_y

    print 'test data dumped successfully.'
    print 'Done!'

    for f in the_files:
        f.close()
        
