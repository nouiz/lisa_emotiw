from birnn import biRNN
from SGD import SGD
from mainLoop import MainLoop
from audio_data import DenseSequences, ListSequences
import numpy
import theano
import theano.tensor as TT

def jobman(state, channel):
    # load dataset
    _train_data = ListSequences(path=state['path'],
                                pca=state['pca'],
                                subset=state['subset'],
                                which='train',
                                one_hot=False,
                                nbits=32)
    train_data = _train_data.export_dense_format(
        sequence_length=state['seqlen'],
        overlap=state['overlap'])

    valid_data = ListSequences(
        path = state['path'],
        pca=state['pca'],
        subset=state['subset'],
        which='valid',
        one_hot=False,
        nbits=32)
    model = biRNN(
        nhids=state['nhids'],
        nouts=numpy.max(train_data.data_y)+1,
        nins=train_data.data_x.shape[-1],
        activ = TT.nnet.sigmoid,
        seed = state['seed'],
        bs = state['bs'],
        seqlen = state['seqlen'])

    algo = SGD(model, state, train_data)

    main = MainLoop(train_data,valid_data, None, model, algo, state, channel)
    main.main()

if __name__=='__main__':
    state = {}

    state['path'] = '/data/lisa/data/faces/EmotiW/complete_audio_features'
    state['bs']  = 128
    state['pca'] = True
    state['subset'] = 'full'
    state['seqlen'] = 50
    state['overlap'] = 30

    state['nhids'] = 100

    state['loopIters'] = 6000
    state['timeStop'] = 32*60
    state['minerr'] = 1e-5

    state['lr'] = .1
    state['lr_adapt'] = 0

    state['seed'] = 123

    state['profile'] = 0

    state['trainFreq'] = 1
    state['validFreq'] = 20
    state['saveFreq'] = 20

    state['prefix'] = 'conv_'
    state['overwrite'] = 1
    jobman(state, None)

