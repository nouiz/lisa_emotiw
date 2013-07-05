import numpy
from pylearn2.utils.iteration import resolve_iterator_class


class EmotiwArrangerIter(object):
    """
    Iterator for the data arranger class
    """
    _default_seed = (17, 2, 946)

    def __init__(self,
                 inst,
                 iter_mode=None,
                 batch_size=None,
                 topo=None,
                 targets=True,
                 rng=_default_seed):

        self.inst = inst
        self.n_face_per_batch = [int(batch_size * self.weights[i]) for i in xrange(len(self.weights))]
        self.batch_size = batch_size

        self.iterators = [inst.dataset[i].iterator(mode=iter_mode,
                                                   batch_size=self.n_face_per_batch[i],
                                                   topo=topo,
                                                   targets=targets) for i in xrange(len(inst.dataset))]
        self.rng = numpy.random.RandomState(rng)

        self.total_n_exs = self.inst.total_n_exs
        self.topo = topo
        self.targets = targets
        self.iter_mode = self.inst.subset_iterator

        self.img_idx = [0]*len(self.inst.datasets)

    @staticmethod
    def fisher_yates_shuffle(imgs, tgts):
        length = imgs.shape[0]
        for i in xrange(length):
            j = numpy.random.randint(0, length)
            imgs[i], imgs[j] = imgs[j], imgs[i]
            tgts[i], tgts[j] = tgts[j], tgts[i]
        return imgs, tgts

    @staticmethod
    def _mix_faces(imgs, tgts):
        #TODO: verify correctness. The original shuffled tgts and returned
        #face_tgts.
        face_imgs = numpy.concatenate(tuple(imgs))
        face_tgts = numpy.concatenate(tuple(tgts))
        EmotiwArrangerIter.fisher_yates_shuffle(face_imgs, face_tgts)
        return face_imgs, face_tgts

    def _pick_idx_given_rnd(self, rnd, weights, num_left):
        cumul_sum = [sum(weights[0:i+1]) * int(bool(num_left[i])) for i in xrange(len(weights))]
        #0 if none left, cumulative sum otherwise.

        #return the index if:
        # - the cumulative sum is larger than rnd (we search from left to
        # right)
        # - all the remaining elements are 0
        #
        # In other words, if all the elements from a given set
        # have been selected for this batch, do as if it didn't exist.
        # If this is the last non-0 elements (left), then this is the one we
        # should sample from.
        for idx, x in enumerate(cumul_sum):
            if x > rnd or (idx < len(cumul_sum)-1 and sum(cumul_sum[idx+1:])==0):
                return idx

    def _get_sequence_idx(self, dset, elem_idx):
        len_lst = [max(1, len(dset.get_sequence(seq))-2) for seq in xrange(len(dset))]
        cumul_sum = [sum(len_lst[:i+1]) for i in xrange(len(len_lst))]

        for idx, tsum in enumerate(cumul_sum):
            if tsum > elem_idx:
                return (idx, cumul_sum[max(0, idx-1)])

    def next(self):
        images, targets = [], []
        next_index = self.iter_mode.next()
        batch_idx = [0]*len(self.inst.datasets)

        die_values = self.rng.rand(self.batch_size)

        for i in xrange(next_index.start, next_index.stop):
            die_value = die_values[sum(batch_idx)]
            pick_from = self._pick_idx_given_rnd(die_value, self.weights,
                    numpy.asarray(self.n_face_per_batch) - numpy.asarray(batch_idx))
            batch_idx[pick_from] += 1
            self.img_idx[pick_from] += 1

            dset = self.inst.datasets[pick_from]
            elem_idx = self.img_idx[pick_from] % len(dset)
            #if the weights are set such that we want more of a given
            #dset than is available, the index will wrap around
            #for the given dset to continue picking data from it.

            the_vals = None

            if hasattr(dset, 'get_sequence'):
                seq_idx, prev_sum = self._get_sequence_idx(dset, elem_idx)
                img_idx = elem_idx - prev_sum
                sequence = dset.get_sequence(seq_idx)
                missing_frames = 3 - (len(sequence)-img_idx)
                the_img = []
                if missing_frames > 0:
                    for i in xrange(img_idx, len(sequence)):
                        the_img.append(sequence.get_original_image(i).tostring())
                    for i in xrange(missing_frames):
                        the_img.append(the_img[-1])
                else:
                    the_img = [sequence.get_original_image(i).to_string() for i in
                                (img_idx-1, img_idx, img_idx+1)]

                the_vals = (map(lambda img: [ord(x) for x in img],
                                    the_img),
                                    sequence.get_7emotion_index(0))
            else:
                the_str = [ord(y) for y in dset.get_original_image(elem_idx).tostring()]
                the_vals = ([the_str for x in range(3)], dset.get_7emotion_index(pick_from))

            images.append(the_vals[0])
            targets.append(the_vals[1])

        images = numpy.asarray(images)
        targets = numpy.asarray(targets)
        return images, targets

    def __iter__(self):
        return self


class ArrangementGenerator(object):
    """
    This generator takes N dataset objects, and combines them offline.
    """
    def __init__(self,
                 datasets,
                 weights):

        assert len(weights) == len(datasets)

        self.datasets = datasets
        total_weight = float(sum(weights))
        self.weights = [w/total_weight for w in weights]
        self.ex_per_dset = []
        self.total_n_exs = 0

        for dset in datasets:
            if hasattr(dset, 'get_sequence'):
                self.ex_per_dset.append(0)
                for seq in xrange(len(dset)):
                    self.ex_per_dset[-1] += max(1, len(dset.get_sequence(seq)) - 2)
                    # images grouped by 3 frames, with overlap. [XOOOOOX] is the range
                    # of valid positions that can yield a frame for times t, t-1 and t+1.
                    # Should there be less than 3 frames available, missing frames will
                    # be generated by copying the last available frame.
            else:
                self.ex_per_dset.append(len(dset))

        self.total_n_exs = sum(self.ex_per_dset)

    def iterator(self,
                 mode=None,
                 batch_size=None,
                 num_batches=None,
                 rng=None):
        """
        Method inherited from the Dataset.
        """
        self.mode = mode
        self.batch_size = batch_size
        mode = resolve_iterator_class(mode)

        self.subset_iterator = mode(self.total_n_exs,
                                    batch_size,
                                    num_batches,
                                    rng=None)

        return EmotiwArrangerIter(self,
                                  mode,
                                  batch_size=batch_size)

    def dump_to(self, path):
        out_X = numpy.memmap(path, mode='w+', dtype=numpy.uint8, shape=(self.total_n_exs, 3, 96, 96, 3))
        out_y = numpy.memmap(path, mode='w+', dtype=numpy.uint8, shape=(self.total_n_exs, 1))

        for idx, item in enumerate(self.iterator()):
            out_X[idx, :] = item[0]
            out_y[idx, :] = item[1]

        del out_X
        del out_y
