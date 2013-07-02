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

    total_sizes = numpy.asarray(train_sizes) + numpy.asarray(test_sizes)
    out_train_x = numpy.memmap(save_as + '_train_x.npy', mode='write', shape=(sum(total_sizes), 96*96*3))

    idx_array = range(sum(total_sizes))
    numpy.random.shuffle(idx_array)

    flush_delay_in_lines = FLUSH_DELAY_IN_LINES

    print 'beginning dump of train data.'

    for i, idx in enumerate(idx_array):
        if flush_delay_in_lines <= 0:
            flush_delay_in_lines = FLUSH_DELAY_IN_LINES
            out_train_x.flush()
    
        the_file = None
        t_size = 0
        for f_idx, x in enumerate(total_sizes):
            if t_size + x > idx:
                the_file = the_files[f_idx]
                break
            else:
                t_size += x
    
        the_path = the_file.root.train.img
        the_idx = idx - t_size
        if train_sizes[f_idx] <= the_idx:
            the_path = the_file.root.test.img
            the_idx = idx - t_size - train_sizes[f_idx]

        char_arr_image = [len(x) != 0 and ord(x) or 0 for x in the_path[the_idx]]

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
    out_train_y = numpy.memmap(save_as + '_train_y.npy', dtype=numpy.float32, mode='write', shape=(sum(total_sizes), len(keypoints_names), 2))
    for i, idx in enumerate(idx_array):
        if flush_delay_in_lines <= 0:
            flush_delay_in_lines = FLUSH_DELAY_IN_LINES
            out_train_y.flush()
        the_file = None
        t_size = 0
        for f_idx, x in enumerate(total_sizes):
            if t_size + x > idx:
                the_file = the_files[f_idx]
                break
            else:
                t_size += x

        the_path = the_file.root.train.label
        the_idx = idx - t_size
        if train_sizes[f_idx] <= the_idx:
            the_path = the_file.root.test.label
            the_idx = idx - t_size - train_sizes[f_idx]

        out_train_y[i, :] = numpy.asarray(the_path[the_idx])
        flush_delay_in_lines -= 1

    del out_train_y
    print 'Done!'

    for f in the_files:
        f.close()
        
