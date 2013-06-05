import os

import numpy as np
import theano

from circular_iterator import circle_around

from cascade_ensembles import CascadeMemberProps, ProbMode, Detector, Ensemble
from output_map import OutputMapFile, OutputMap, CroppedPatchesDataset

from pylearn2.utils.iteration import SequentialSubsetIterator

class MultiScaleCascadedDetectorEnsembles(Ensemble, Detector):
    """
    Cascaded ensembles class.

    Parameters:
    ---------
        n_members: Number of predictors in the ensemble.
        init_reject_probability: The reject probability of the first predictor.
        prob_gen_mode: The mode to generate the initial reject probabilities of the cascade
        architecture.
        reject_probabilities: The prespecified reject probabilities of the cascade.
        img_shape: shape of the image.
        non_max_sup: Parametetr for non-maximum suppression.
        output_map_shp: The shape of the output map. 3D tensor that has the dimensions: (x, y, batch_size)
    """
    def __init__(self,
            n_members,
            init_reject_probability=0.9,
            prob_gen_mode=ProbMode.LIN_INTERP,
            reject_probabilities=None,
            nlevels=2,
            img_shape = None,
            non_max_radius = None,
            output_map_shp=None):

        self.n_members = n_members
        self.reject_probabilities = []
        self.predictors = []
        self.training_algos = []
        self.output_map_shp = output_map_shp
        self.non_max_radius = non_max_radius
        self.nlevels = nlevels

        assert output_map_shp is not None, ("Output map should not be left empty.")
        assert init_reject_probability <= 1.0, ("Initial reject probability should be less than or equal to 1.")

        if prob_gen_mode == ProbMode.LIN_INTERP:
            diff = 0.9999 - init_reject_probability
            prob_inc = diff / float(n_members)
            for i in xrange(n_members):
                self.reject_probabilities.append(prob_inc)
        elif prob_gen_mode == ProbMode.CONSTANT:
            for i in xrange(n_members):
                self.reject_probabilities.append(init_reject_probability)
        else:
            raise Exception("Invalid cascade probability generation mode is selected.")

        self.preds_size = np.prod(output_map_shp[0:2])

        dop = np.arange(self.preds_size)
        dop = dop.reshape((dop.shape[0], 1))
        self.data_predictor_positions = dop
        self.setup_output_map(output_map_shp=output_map_shp)
        self.img_shape = img_shape

    def setup_output_map(self, output_map_shp=None):
        """
        Setup the convolutional output map of the neural network.

        Parameters
        ----------
            output_map_shp: list,
            The shape of output map of the neural net.
        """
        if output_map_shp is None:
            raise Exception("Shape of the output map should not be empty.")
        self.output_map_shp = output_map_shp

    def add_predictors(self, predictor):
        """
        Add a new predictor to the class.

        Parameters
        -----------
            predictor: MultiScaleConvnetMLP object
        """
        assert predictor is not None
        assert predictor.nlevels == self.nlevels
        self.predictors.append(predictor)

    def get_predictor_prediction(self, n_predictor_no, dataset):
        """
        Get the prediction of the predictor.

        Parameters:
        ----------
            n_predictor_no: The predictor's index.
            dataset: The dataset class for the training.
        """
        assert dataset.X is not None
        assert self.predictors[n_predictor_no] is not None, ("The requested predictor couldn't be found.")
        return self.predictors[n_predictor_no].get_posteriors(dataset)

    def classify(self, dataset):
        """
        Perform hard classificatio for the example in the cascade.
        """
        assert dataset.X is not None
        predictions = self.perform_detection(dataset)

        results = np.all(predictions==0., axis=0)
        classifications = []

        for res in results:
            if res:
                classifications.append(0)
            else:
                classifications.append(1)

        return classifications

    def train_members(self, datasets=None):
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

        assert datasets is not None, "Dataset should not be empty"
        assert len(datasets) == self.nlevels

        posteriors_files = None
        size_of_rf = self.predictors[0].receptive_field_size

        for i in xrange(self.n_members):
            if i == 0:
                self.predictors[i].model._setup_trainers(datasets)
                self.predictors[i].train()
                posteriors_files = self.predictors[0].get_posterior_patches(datasets,
                        cascade_no=i, size_of_rf=size_of_rf)
            else:
                datasets = []
                for posterior_file in posteriors_files:
                    cropped_patches_ds = CroppedPatchesDataset(img_shape=size_of_rf,
                            h5_file=posterior_file)
                    datasets.append(cropped_patches_ds)

                self.predictors[i].model._setup_trainers(datasets)
                self.predictors[i].train()
                posteriors_files = self.predictors[i].get_posterior_patches(datasets,
                        cascade_no=i, size_of_rf=size_of_rf)

    def perform_detection(self, datasets):
        """
        Perform the detection on the images.
        Similar to training at each level of the cascade
        perform thresholding and non-maximum suppression.
        """
        assert datasets is not None, "Datasets should not be empty."

        assert len(datasets) == self.nlevels

        posteriors_files = None
        size_of_rf = self.predictors[0].receptive_field_size

        for i in xrange(self.n_members):
            if i == 0:
                posteriors_files = self.predictors[0].get_posterior_patches(datasets,
                        cascade_no=i, size_of_rf=size_of_rf)
            else:
                datasets = []
                for posterior_file in posteriors_files:
                    cropped_patches_ds = CroppedPatchesDataset(img_shape=size_of_rf,
                            h5_file=posterior_file)
                    datasets.append(cropped_patches_ds)

                posteriors_files = self.predictors[i].get_posterior_patches(datasets,
                        cascade_no=i, size_of_rf=size_of_rf)

        return posteriors_files

    def _check_threshold_non_max_suppression(self, probs, threshold, radius):
        """
        Check the non maximum supression with threshold.
        Explanation:
            Karim proposed a precise procedure which is standard for this: sort the outputs
        (for different locations in the same image) in decreasing order of probability (keeping
        only those above a threshold). Traverse that list from highest face probability down.
        For each entry, remove the entries below that correspond to locations that are too
        close spatially (corresponding to more than 50% overlap of the bounding boxes).
        """
        assert self.img_shape is not None
        assert radius < self.img_shape[0]

        batch_size = probs.shape[0]
        processed_probs = []

        for i in xrange(batch_size):
            dop = np.copy(self.data_predictor_positions)
            #Make the predictions a column vector
            preds = np.reshape(probs[i], (probs[i].shape[0], 1))
            pred_pos = np.hstack((dop, preds))
            pred_pos = pred_pos[pred_pos[:,1].argsort()]
            sorted_preds = pred_pos[:, 1]

            new_preds = np.zeros(preds.shape)

            for j in xrange(self.preds_size):
                #preds_view = preds.reshape(self.output_map_shp[:2])
                if sorted_preds[j] < threshold:
                    new_preds[pred_pos[j,0]] = 0
                else:
                    #Remove the spatially close entries, hence check the, check the neighbours
                    #within radius neighbourhood.

                    #The problem is that the spatially close outputs might be corresponding to the same face.
                    #Rows:
                    r = j % self.output_map_shp[1]

                    #Columns:
                    c = j - r * self.output_map_shp[1]
                    iterator = circle_around(r, c)

                    #Move on the grid in a circular fashion
                    for loc in iterator:
                        if loc[0] == radius:
                            break
                        y, x = loc[1]
                        n = y * self.output_map_shp[0] + self.output_map_shp[1]
                        if (loc[1][0] >= 0 and loc[1][0] <= self.img_shape[0] and loc[1][1] <= self.img_shape[1] and loc[1][1] >= 0):
                            if (new_preds[pred_pos[n, 0]] < new_preds[pred_pos[j, 0]]):
                                new_preds[pred_pos[n, 0]] = 0
                processed_probs.append(new_preds)
            processed_probs = np.asarray(processed_probs)
        return processed_probs

    def _check_non_max_suppression(self, probs):
        """
        Check only the non maximum supression.
        """
        batch_size = self.output_map_shp[-1]
        for i in xrange(batch_size):
            dop = np.copy(self.data_predictor_positions)
            preds = np.reshape(dop, (probs[i].shape[0], 1))
            pred_pos = np.vstack((dop, preds))
            pred_pos = pred_pos[pred_pos[:,1].argsort()]

    def _check_threshold(self, outputs, threshold):
        """
        Check the threshold for the images
        """
        outputs[outputs < threshold] = 0
        return outputs

    def create_output_map(self, out_map, img_loc):
        conv_out_map = np.zeros(self.output_map_shp)
        i = 0

        for out_val in out_map:
            conv_out_map[img_loc[i]] = out_val
            i += 1
        return conv_out_map

    def save_posterior_patches(self, datasets, cascade_no, size_of_rf=None):
        """
        Save the information about the cascade and the fprop to here.
        """
        mode = SequentialSubsetIterator
        targets = True

        count_patches = 0
        out_files = []
        model = self.predictors[cascade_no]

        for i in xrange(self.nlevels):

            name = "cascade_%d_lvl_%d" % (cascade_no, i)
            input_space = model.models[i].get_input_space()
            X = input_space.make_theano_batch()

            # doesn't support topo yets
            if X.ndim > 2:
                assert False

            dataset = datasets[i]

            if dataset.iter_mode == "train":
                dataset.set_iter_mode("fprop")

            outputs = model.fprop(X, cascade_no)
            batch_size = model.batch_size
            fn = theano.function([X], outputs)

            n_examples = X.shape[0]
            model_file_path = model.model_file_path
            receptive_field_size = model.receptive_field_size

            out_file, gcols = OutputMapFile.create_file(model_file_path,
                    name, n_examples=n_examples, out_shape=(n_examples,
                        size_of_rf[0], size_of_rf[1]))

            outmap = OutputMap(receptive_field_size, self.img_shape, model.stride)

            count_patches = 0
            #This is a counter for the cascade i where i > 0.
            img_no = 0
            for data in dataset.iterator(batch_size=batch_size, mode=mode, targets=True):
                if model.isconv:
                    #Extract the patches from the full size image according to the convolution
                    #operation with respect to a specific sized receptive fields
                    #and stride.
                    minibatch_images = data[0]
                    conv_targets = data[1]

                    batch_act = fn(minibatch_images)
                    batch_act = np.concatenate([elem.reshape(elem.size) for elem in batch_act], axis=0)

                    new_preds = self._check_threshold_non_max_suppression(batch_act,
                            self.reject_probabilities[cascade_no],
                            self.non_max_radius)

                    start = count_patches

                    patches, targets, img_locs, img_nos = outmap.extract_patches_batch(
                            minibatch_images,
                            new_preds,
                            start,
                            conv_targets)

                    stop = count_patches + patches.shape[0]

                    OutputMapFile.save_output_map(out_file,
                            patches,
                            new_preds,
                            img_nos,
                            targets=True,
                            start=start,
                            stop=stop)

                    count_patches = stop
                else:
                    # Do the patchwise patch extraction
                    # Decide with patches to send to the next cascade member
                    minibatch_patches = data[0]
                    minibatch_targets = data[1]
                    minibatch_imgnos = data[2]
                    minibatch_imglocs = data[3]

                    batch_act = fn(minibatch_patches)
                    batch_act = np.concatenate([elem.reshape(elem.size) for elem in batch_act], axis=0)

                    output_map = self.create_output_map(minibatch_targets, minibatch_imglocs)

                    new_preds = self._check_threshold_non_max_suppression(output_map,
                            self.reject_probabilities[cascade_no],
                            self.non_max_radius)

                    patches, targets, img_locs, img_nos = outmap.get_next_patches(
                            minibatch_patches,
                            new_preds,
                            minibatch_imgnos,
                            conv_targets)

                    start = count_patches
                    stop = count_patches + patches.shape[0]

                    OutputMapFile.save_output_map(out_file,
                            minibatch_patches,
                            new_preds,
                            img_nos,
                            targets=True,
                            start=start,
                            stop=stop)

                    count_patches = stop

            h5_path = os.path.join(model_file_path, name)
            out_files.append(h5_path)

            #out_files.append(out_file)
        return out_files

class MultiscaleConvolutionalCascadeMemberTrainer(object):
    """
    Multiscale convolutional cascade trainer class.

    Parameters:
    ----------
        sparsity_level: The sparsity level for the convolutional training.
        receptive_field_size: The size of the receptive field.
        save_path: The path to save the pylearn2 pickle files.
        train_extensions: Pylearn2 training extensions
    """
    def __init__(self,
            cascade_no,
            sparsity_level=0.5,
            isconv=True,
            receptive_field_size=None,
            model_file_path=None,
            stride=None,
            use_sparsity=False,
            model=None):

        self.isconv = isconv
        self.use_sparsity = use_sparsity
        self.sparsity_level = sparsity_level
        self.stride = stride
        self.model_file_path = model_file_path

        self.receptive_field_size = receptive_field_size
        self.model = model
        self.cascade_no = cascade_no

    def set_model(self, model):
        """
        Set the default model and the algorithm of the trainer class.
        """
        self.model = model

    """
    def facemap_train(self, facemap=None):
        #Train the convolutional cascade member.
        def check_sparsity(facemap=None):
            #Forward prop. If the facemap is sparser than a certain value, then don't
            #do fully convolutional training.
            if facemap is None:
                raise Exception("Facemap shouldn't be empty.")
            sparse_vals = np.where(facemap==0.)
            n_sparse_vals = sparse_vals[0].shape[0]
            n_vals = facemap.shape[0]
            sparsity_level = float(n_sparse_vals) / float(n_vals)
            return sparsity_level

        #TODO, check that code again!.
        sparsity_lvl = check_sparsity(facemap)
        if self.use_sparsity:
            if sparsity_lvl <= self.sparsity_level:
                self.train()
            else:
                r, c = facemap.shape
                for i in xrange(r):
                    for j in xrange(c):
                        if facemap[r, c] != 0.:
                            #Dataset class should have a set_loc class that iterator will
                            #only return the set_loc patches and labels extracted from the grid.
                            self.model.dataset.set_loc(r, c)
                            self.trainer.dataset = self.dataset
                            self.trainer.main_loop()
        else:
            r, c = facemap.shape
            for i in xrange(r):
                for j in xrange(c):
                    if facemap[r, c] != 0.:
                        self.dataset.set_loc(r, c)
                        self.trainer.dataset = self.dataset
                        self.trainer.main_loop()
    """

    def train(self):
        """
        Train the cascade on the whole image.
        """
        self.model.train_mscale()

    def get_posteriors(self, dataset):
        """
        Return the posterior probabilities of the cascadeMember. This returns
        the P(Face|x).
        """
        acts = []
        for i in xrange(self.model.nlevels):
            input_space = self.model.models[i].get_input_space()
            X = input_space.make_theano_batch()
            # doesn't support topo yets
            if X.ndim > 2:
                assert False

            outputs = self.model.fprop(X, i)
            batch_size = self.model.batch_size
            fn = theano.function([X], outputs)
            act = []

            for i in xrange(0, X.shape[0], batch_size):
                batch = X[i:i+batch_size,:]
                batch_act = fn(batch)
                batch_act = np.concatenate([elem.reshape(elem.size) for elem in batch_act],axis=0)
                act.append(batch_act)

            act = np.concatenate(act, axis=0)
            acts.append(act)
        return acts

