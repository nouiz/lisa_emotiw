import numpy as np
import theano
import theano.tensor as T

from circular_iterator import circle_around
from pylearn2.train import Train

class ProbMode(object):
    LIN_INTERP = "LINEAR_INTERPOLATE"
    CONSTANT = "CONSTANT"

class Detector(object):
    """
    Base class for the detectors.
    """
    def perform_detection(self, dataset):
        pass

class Ensemble(object):
    """
    Base class for Ensemble training, like boosting, ...etc.
    """
    def __init__(self, n_predictors, reject_probability):
        pass

    """
    Add a new predictor to the dictionary
    """
    def add_predictor(self, predictor):
        pass

    """
    Get the prediction of the ensemble
    """
    def get_prediction(self):
        pass

    """
    Train the ensembles.
    """
    def train_predictors(self):
        pass

class CascadedDetectorEnsembles(Ensemble, Detector):
    """
    Cascaded ensembles class.

    Parameters:
    ---------
    n_predictors: Number of predictors in the ensemble.
    init_reject_probability: The reject probability of the first predictor.
    prob_gen_mode: The mode to gene.
    output_map_shp: The shape of the output map. 3D tensor that has the dimensions: (x, y, batch_size)
    """
    def __init__(self,
            n_predictors,
            init_reject_probability=0.9,
            prob_gen_mode=ProbMode.LIN_INTERP,
            reject_probabilities=None,
            img_shape = None,
            non_max_radius = None,
            output_map_shp=None):

        self.n_predictors = n_predictors
        self.reject_probabilities = []
        self.predictors = []
        self.training_algos = []
        self.output_map_shp = output_map_shp
        self.non_max_radius = non_max_radius

        assert output_map_shp is not None, ("Output map should not be left empty.")
        assert init_reject_probability <= 1.0, ("Initial reject probability should be less than or equal to 1.")

        if prob_gen_mode == ProbMode.LIN_INTERP:
            diff = 0.9999 - init_reject_probability
            prob_inc = diff / float(n_predictors)
            for i in xrange(n_predictors):
                self.reject_probabilities.append(prob_inc)
        elif prob_gen_mode == ProbMode.CONSTANT:
            for i in xrange(n_predictors):
                self.reject_probabilities.append(init_reject_probability)
        else:
            raise Exception("Invalid cascade probability generation mode is selected.")
        self.preds_size = np.prod(output_map_shp[0:2])
        dop = np.arange(self.preds_size)
        dop = dop.reshape((dop.reshape[0], 1))
        self.data_predictor_positions = dop
        self.setup_output_map(output_map_shp=output_map_shp)
        self.img_shape = img_shape

    """
    Setup the convolutional output map of the neural network.

    Parameters
    ----------
    output_map_shp: list
    The shape of output map of the neural net.
    """
    def setup_output_map(self, output_map_shp=None):
        if output_map_shp is None:
            raise Exception("Shape of the output map should not be empty.")
        pre_output_map = np.ones(output_map_shp)

    """
    Add a new predictor to the class.

    Parameters
    -----------
    predictor: Predictor
    train_algo: Training algorithm of the ensemble.
    """
    def add_predictor(self, predictor, train_algo):
        assert predictor is not None
        assert train_algo is not None
        self.predictors.append(predictor)
        self.training_algos.append(train_algo)

    """
    Get the prediction of the predictor.

    Parameters:
    ----------
    n_predictor_no: The predictor's index.
    dataset: The dataset class for the training.
    """
    def get_predictor_prediction(self, n_predictor_no, dataset):
        assert dataset.X is not None
        assert self.predictors[n_predictor_no] is not None, ("The requested predictor couldn't be found.")
        return self.predictors[n_predictor_no].get_posteriors(dataset)

    """
    Classify the instance in the cascade.
    """
    def classify(self, dataset):
        assert dataset.X is not None
        predictions = self.perform_detection(dataset)
        X = dataset.X
        results = np.all(X==0., axis=0)
        classifications = []
        for res in results:
            if res:
                classifications.append(-1)
            else:
                classifications.append(1)
        return classifications

    """
    Train the predictors.
    You need to do non-maximal suppression for training as well, as follows:
    - every learner sees ALL the positive examples from the training set, but
    - only the negative examples that have not been confidently detected by the previous predictors arrive to predictor
    So if the prob(face|x)<=threshold reject.
    and
    - for every positive example bounding box, we can generate negative examples from the nearby bounding
    boxes (in the image from which they come from) that do not overlap by more than 50% (intersection / union of boxes)
    """
    def train_predictors(self, dataset=None):
        assert dataset is not None, "Dataset should not be empty"
        radius = self.non_max_radius
        for i in xrange(self.n_predictors):
            if i == 0:
                self.predictors[i].train(dataset.X, dataset.y)
            else:
                reject_prob = self.reject_probabilities[i]
                posteriors = self.predictors[i-1].get_posteriors(dataset)
                face_map = self._check_threshold_non_max_suppression(posteriors,
                        threshold=reject_prob, radius=radius)
                self.predictors[i].train(dataset.X, dataset.y, facemap=face_map)

    """
    Perform the detection on the images. Similar to training at each level of the cascade
    perform thresholding and non-maximum suppression.
    """
    def perform_detection(self, dataset):
        assert dataset is not None
        radius = self.non_max_radius
        for i in xrange(self.n_predictors):
            if i == 0:
                self.predictors[i].train(dataset.X, dataset.y)
            else:
                reject_prob = self.reject_probabilities[i]
                posteriors = self.predictors[i-1].get_posteriors(dataset)
                face_map = self._check_threshold_non_max_suppression(posteriors,
                        threshold=reject_prob, radius=radius)
                posteriors = self.predictors[i].get_posteriors(dataset.X, dataset.y, facemap=face_map)
        return posteriors

    """
    Check the non maximum supression with threshold.
    Explanation:
        Karim proposed a precise procedure which is standard for this: sort the outputs (for
        different locations in the same image) in decreasing order of probability (keeping only those
        above a threshold). Traverse that list from highest face probability down.
        For each entry, remove the entries below that correspond to locations that are too
        close spatially (corresponding to more than 50% overlap of the bounding boxes).

    """
    def _check_threshold_non_max_suppression(self, probs, threshold, radius):
        assert self.img_shape is not None
        assert radius < self.img_shape[0]

        batch_size = self.output_map_shp[-1]
        processed_probs = []

        for i in xrange(batch_size):
            dop = np.copy(self.data_predictor_positions)
            preds = np.reshape(probs[i], (probs[i].shape[0], 1))
            pred_pos = np.vstack((dop, preds))
            pred_pos = pred_pos[pred_pos[:,1].argsort()]
            sorted_preds = pred_pos[:, 1]

            border = 0
            new_preds = np.zeros()
            for j in xrange(self.preds_size):
                preds_view = preds.reshape(self.output_map_shp[:2])
                if sorted_preds[j] < threshold:
                    new_preds[pred_pos[j,0]] = 0
                else:
                    #Remove the spatially close entries, hence check the, check the neighbours
                    #within radius neighbourhood.
                    #The problem is that the spatially close outputs might be corresponding to the same face.
                    #Rows:
                    r = j % self.output_map_shp[1]
                    #Columns:
                    c = j - r*self.output_map_shp[1]
                    iterator = circle_around(r, c)
                    #Move on the grid in a circular fashion
                    for loc in iterator:
                        if loc[0] == radius:
                            break
                        y, x = loc[1]
                        n = y * self.output_map_shp[0] + self.output_map_shp[1]
                        if (loc[1][0] >= 0 and loc[1][0] <= self.img_shape[0] and
                                loc[1][1] <= self.img_shape[1] and loc[1][1] >= 0):
                            if new_preds[pred_pos[n, 0]] < new_preds[pred_pos[j, 0]]:
                                new_preds[pred_pos[n, 0]] = 0
            processed_probs = new_preds
        return new_preds

    """
    Check the non maximum supression.
    """
    def _check_non_max_suppression(self, probs):
        batch_size = self.output_map_shp[-1]
        for i in xrange(batch_size):
            dop = np.copy(self.data_predictor_positions)
            preds = np.reshape(dop, (probs[i].shape[0], 1))
            pred_pos = np.vstack((dop, preds))
            pred_pos = pred_pos[pred_pos[:,1].argsort()]

    """
    Check the threshold for the images
    """
    def _check_threshold(self, outputs, threshold):
        outputs[outputs < threshold] = 0
        return outputs

class ConvolutionalCascadeMemberTrainer(object):

    """
    Convolutional cascade trainer class.

    Parameters:
    ----------
    sparsity_level: The sparsity level for the convolutional training.
    receptive_field_size: The size of the receptive field.
    save_path: The path to save the pylearn2 pickle files.
    train_extensions: Pylearn2 training extensions
    """
    def __init__(self, sparsity_level=0.5, receptive_field_size=None, save_path=None,
            train_extensions=None):
        self.model = None
        self.algorithm = None
        self.sparsity_level = sparsity_level
        self.receptive_field_size = receptive_field_size
        self.save_path = save_path
        self.trainer = None
        self.train_extensions = None

    """
    Set the default model and the algorithm of the trainer class.
    """
    def set_model(self, model, algorithm, trainset):
        self.model = model
        self.algorithm = algorithm
        if self.save_path is None:
            self.save_path = model.name + ".pkl"
        self.dataset = trainset
        self.trainer = Train(model = self.model, algorithm = algorithm,
                save_path=self.save_path, save_freq=1,
                extensions = self.train_extensions, dataset = trainset)

    """
    Train the convolutional cascade member.
    """
    def facemap_train(self, facemap=None):

        """
            Forward prop. If the facemap is sparser than a certain value, then don't
        do fully convolutional training.
        """
        def check_sparsity(facemap=None):
            if facemap is None:
                raise Exception("Facemap shouldn't be empty.")
            sparse_vals = np.where(facemap==0.)
            n_sparse_vals = sparse_vals[0].shape[0]
            n_vals = facemap.shape[0]
            sparsity_level = float(n_sparse_vals) / float(n_vals)
            return sparsity_level
        sparsity_lvl = check_sparsity(facemap)
        if sparsity_lvl <= self.sparsity_level:
            self.train()
        else:
            r, c = facemap.shape
            for i in xrange(r):
                for j in xrange(c):
                    if facemap[r,c] != 0.:
                        #Dataset class should have a set_loc class that iterator will
                        #only return the set_loc patches and labels extracted from the grid.
                        self.dataset.set_loc(r,c)
                        self.trainer.dataset = self.dataset
                        self.trainer.main_loop()
    """
    Train the cascade on the whole image.
    """
    def train(self):
        self.trainer.main_loop()

    """
    Perform forward prop.
    """
    def fprop(self, X):
        below = X
        outputs = []
        for layer in self.model.layers:
            above = layer.fprop(below)
            below = above
            outputs.append(above)
        assert len(outputs) > 0
        return [outputs]

    """
        Return the posterior probabilities of the cascadeMember. This returns the
    P(Face|x).
    """
    def get_posteriors(self, dataset):
        input_space = self.model.get_input_space()
        X = input_space.make_theano_batch()
        if X.ndim > 2:
            assert False # doesn't support topo yets
        outputs = self.fprop(X)
        batch_size = self.model.batch_size
        fn = theano.function([X], outputs)
        act = []
        for i in xrange(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size,:]
            batch_act = fn(batch)
            batch_act = np.concatenate([elem.reshape(elem.size) for elem in batch_act],axis=0)
            act.append(batch_act)
        act = np.concatenate(act, axis=0)
        return act

