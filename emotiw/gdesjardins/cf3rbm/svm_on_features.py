import argparse
import copy
import numpy
import theano
import pickle
import logging
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.preprocessing import Standardize
from pylearn2.training_callbacks.training_callback import TrainingCallback
from scikits.learn.svm import LinearSVC

logging.basicConfig(level=logging.INFO)

class pylearn2_svm_callback(TrainingCallback):

    def __init__(self, run_every=1, results_prefix='', **kwargs):
        self.run_every = run_every
        self.results_prefix = results_prefix
        self.svm_on_features = SVMOnFeatures(**kwargs)
    
    def __call__(self, model, dataset, algorithm):
        if model.batches_seen == 0 or (model.batches_seen % self.run_every) != 0:
            return

        i = model.batches_seen
        best_svm, (train_error, valid_error, test_error) = self.svm_on_features.run()
        i = model.batches_seen

        results = {}
        results['%sbatches_seen' % self.results_prefix] = i
        results['%strerr_%i' % (self.results_prefix, i)] = train_error
        results['%svalerr_%i' % (self.results_prefix, i)] = valid_error
        results['%ststerr_%i' % (self.results_prefix, i)] = test_error
        results['%sC%i' % (self.results_prefix, i)] = best_svm.C

        fp = open('svm_callback.log', 'a')
        fp.write('Batches seen: %i' % i)
        fp.write('\t best validation error: %f' % valid_error)
        fp.write('\t best test error: %f' % test_error)
        fp.write('\t best svm.C: %f' % best_svm.C)
        fp.write('\n')
        fp.close()

        if hasattr(model, 'jobman_channel') and model.jobman_channel:
            model.jobman_state.update(results)
            model.jobman_channel.save()

def is_onehot(y):
    return y.ndim == 2 and (numpy.sum(y, axis=1) == 1).all()

def process_labels(y):
    if is_onehot(y):
        return numpy.argmax(y, axis=1)
    else:
        return y.flatten()

class SVMOnFeatures():
   
    def __init__(self, svm, trainset, testset,
                 model=None, model_call_kwargs=None,
                 validset=None,
                 C_list=None,
                 save_fname=None,
                 retrain_on_valid=True):
        """
        Performs cross-validation of a linear SVM on features extracted by a unsupervised
        learning module.

        Parameters
        ----------
        svm: sklearn.svm
            Scikit-learn SVM object to use for training.
        trainset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for (feature extraction, SVM training).
        testset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for final performance measurement. If no validation set is
            provided, the test will also be used for hyper-parameter selection.
        model: pylearn2.models.model.Model
            Unsupervised learning module used for feature extraction.
        model_call_kwargs: dict (optional)
            Dictionary of arguments to pass to model.call.
        validset: pylearn2.datasets.dataset.Dataset
            Pylearn2 dataset used for hyper-parameter selection.
        C_list: list
            List of penalty parameters to cross-validate for the SVM.
        save_fname: string
            Output filename to store trained svm model.
        """
        self.svm = svm
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.model = model
        #self.model.do_theano()
        self.model_call_kwargs = model_call_kwargs
        if C_list is None:
            C_list = [1e-3,1e-2,1e-1,1,10]
        self.C_list = C_list
        self.save_fname = save_fname
        self.retrain_on_valid=retrain_on_valid
        self.trainset_y = process_labels(self.trainset.y)
        self.testset_y  = process_labels(self.testset.y)
        self.validset_y = process_labels(self.validset.y) if validset else self.testset_y

    def extract_features(self, dset, preproc=None, can_fit=False):
        dset.X = dset.X.astype(theano.config.floatX)
        dset.X = numpy.hstack((dset.X[:-1], dset.X[1:]))
        new_dset = dense_design_matrix.DenseDesignMatrix()
        if  str(self.model.__class__).find('DBM.dbm') != -1:
            self.model.set_batch_size(len(dset.X))
            self.model.setup_pos_func(dset.X)
            self.model.pos_func()
            new_dset.X = self.model.fn(dset.X)
        elif self.model:
            outsize = self.model.fn(dset.X[:1]).shape[1]
            X = numpy.zeros((len(dset.X), outsize))
            for i in xrange(0, len(X), self.model.batch_size):
                batch = dset.X[i : i + self.model.batch_size]
                X[i : i + len(batch)] = self.model.fn(batch)
            new_dset.X = X
        else:
            new_dset.X = dset.X

        if preproc:
            preproc.apply(new_dset, can_fit=True)

        return new_dset
 
    def run(self):
        # Extract features from model.
        preproc = Standardize()
        if self.model:
            self.model.fn = self.model.function("perform", **self.model_call_kwargs)
        newtrain = self.extract_features(self.trainset, preproc, can_fit=True)
        newtest  = self.extract_features(self.testset, preproc, can_fit=False)
        newvalid = newtest if not self.validset else\
                   self.extract_features(self.validset, preproc, can_fit=False)

        # Find optimal SVM hyper-parameters.
        (best_svm, train_error, valid_error) = cross_validate_svm(self.svm,
                (newtrain.X, self.trainset_y),
                (newvalid.X, self.validset_y),
                self.C_list)
        logging.info('Best train/valid error for C=%f : %f \t %f' % (best_svm.C, train_error, valid_error))

        # Optionally retrain on validation set, using optimal hyperparams.
        if self.validset and self.retrain_on_valid:
            retrain_svm(best_svm,
                    (newtrain.X, self.trainset_y),
                    (newvalid.X, self.validset_y))

        test_error = compute_test_error(best_svm, (newtest.X, self.testset_y))
        logging.info('Test error = %f' % test_error)
        if self.save_fname:
            fp = open(self.save_fname, 'w')
            pickle.dump(best_svm, fp)
            fp.close()
        return best_svm, (train_error, valid_error, test_error)


import time
def cross_validate_svm(svm, (train_X, train_y), (valid_X, valid_y), C_list):
    best_svm = None
    best_train_error = numpy.Inf
    best_valid_error = numpy.Inf

    print 'C_list = ', C_list
    for C in C_list:
        print 'C = ', C
        t1 = time.time()
        if hasattr(svm, 'set_params'):
            svm.set_params(C = C)
        else:
            svm.C = C
        
        # Fit training set.
        svm.fit(train_X, train_y)

        # Measure training error.
        train_predy = svm.predict(train_X)
        train_error = (train_y[:-1] != train_predy).mean()
        # Measure validation error.
        valid_predy = svm.predict(valid_X)
        valid_error = (valid_y[:-1] != valid_predy).mean()

        # If beset validation error...
        if valid_error < best_valid_error:
            logging.info('SVM(C=%f): train_error=%f valid_error=%f **' % (C, train_error, valid_error))
            best_train_error = train_error
            best_valid_error = valid_error
            best_svm = copy.deepcopy(svm)
            # Numpy bug workaround: copy module does not respect C/F ordering.
            best_svm.raw_coef_ = numpy.asarray(best_svm.raw_coef_, order='F')
        else:
            logging.info('SVM(C=%f): train_error=%f valid_error=%f' % (C, train_error, valid_error))
        logging.info('Elapsed time: %f' % (time.time() - t1))
    return (best_svm, best_train_error, best_valid_error)


def retrain_svm(svm, (train_X, train_y), (valid_X, valid_y)):
    logging.info('Retraining on {train, validation} sets.')
    full_train_X = numpy.vstack((train_X, valid_X))
    full_train_y = numpy.hstack((train_y, valid_y))
    svm.fit(full_train_X, full_train_y.flatten())
    return svm


def compute_test_error(svm, (test_X, test_y)):
    test_predy = svm.predict(test_X)
    return (test_y[:-1] != test_predy).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train a linear SVM on the features extracted by a given model.")
    parser.add_argument('config', action='store', choices=None,
                        help='A YAML configuration file specifying the training procedure')
    args = parser.parse_args()
    obj = serial.load_train_file(args.config)
    obj.run()
