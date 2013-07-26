import tables
import os
import sys


def split(fname, which_set, start=0, stop=-1):
    f = tables.openFile(fname)
    lgt = f.root.Data.X.shape[0]

    if stop < 0:
        stop = lgt

    assert stop <= lgt
    assert 0 <= start < stop

    prev_x = f.root.Data.X
    prev_y = f.root.Data.y

    filters = tables.Filters(complib='blosc', complevel=5)

    output = tables.openFile(os.path.splitext(fname)[0] + '_' + which_set + '.h5', mode='w', title=which_set)
    data = output.createGroup(output.root, 'Data', 'Data')
    x = output.createCArray(data, 'X', atom=prev_x.atom, shape=(stop-start, 96, 96, 3), title='Data values', filters = filters)
    y = output.createCArray(data, 'y', atom=prev_y.atom, shape=(stop-start, 196, 96), title='Data targets', filters = filters)

    x[:] = prev_x[start:stop,:]
    y[:] = prev_y[start:stop,:]

    f.flush()

if __name__ == '__main__':
    #input filename, which_set, start, stop
    args = sys.argv[1:]
    if len(args) == 2:
        split(args[0], args[1])
    elif len(args) == 3:
        split(args[0], args[1], int(args[2]))
    elif len(args) == 4:
        split(args[0], args[1], int(args[2]), int(args[3]))
    else:
        print 'Incorrect invokation.'
