from sgd import SGD
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace
from theano import config
import theano.tensor as T
from theano import function
from theano.gof.op import get_debug_values
import logging
log = logging.getLogger(__name__)
from pylearn2.utils import sharedX
from pylearn2.utils.timing import log_timing
import theano
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
import numpy
from pylearn2.utils.iteration import is_stochastic


class KeypointADADELTA(TrainingAlgorithm):
    """
    Stochastic Gradient Descent
    WRITEME: what is a good reference to read about this algorithm?

    A TrainingAlgorithm that does gradient descent on minibatches.

    """
    E_g2 = {}
    E_dx2 = {}
    learning_rate = {}
    def __init__(self, 
                 decay_factor = 0.95,
                 cost=None,
                 termination_criterion=None,
                 monitoring_dataset=None,
                 batch_size=None,
                 train_iteration_mode=None,
                 batches_per_iter=None,
                 update_callbacks=None,
                 set_batch_size=None,
                 monitoring_costs=None,
                 theano_function_mode=None,
                 monitoring_batches=None):
        self.decay_factor = decay_factor
        self.cost = cost
        self.batch_size = batch_size
        self._set_monitoring_dataset(monitoring_dataset)
        self._register_update_callbacks(update_callbacks)
        self.rng = np.random.RandomState([2012, 10, 5])

        self.monitoring_batches = monitoring_batches

        if train_iteration_mode is None:
            train_iteration_mode = 'shuffled_sequential'
        self.train_iteration_mode = train_iteration_mode

        if monitoring_dataset is None:
            if monitoring_batches is not None:
                raise ValueError("Specified an amount of monitoring batches but not a monitoring dataset.")

        self.set_batch_size = set_batch_size
        self.batches_per_iter = batches_per_iter

        self.termination_criterion = termination_criterion
        self.monitoring_costs = monitoring_costs
        self.theano_function_mode = theano_function_mode
        self.first = True

    def setup(self, model, dataset):
        inf_params = [ param for param in model.get_params() if np.any(np.isinf(param.get_value())) ]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value())) for param in model.get_params()]):
            nan_params = [ param for param in model.get_params() if np.any(np.isnan(param.get_value())) ]
            raise ValueError("These params are NaN: "+str(nan_params))
        self.model = model

        batch_size = self.batch_size
        if hasattr(model, "force_batch_size"):
            if model.force_batch_size > 0:
                if batch_size is not None:
                    if batch_size != model.force_batch_size:
                        if self.set_batch_size:
                            model.set_batch_size(batch_size)
                        else:
                            raise ValueError("batch_size argument to SGD conflicts with model's force_batch_size attribute")
                else:
                    self.batch_size = model.force_batch_size
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        # TODO: come up with some standard scheme for associating training runs
        # with monitors / pushing the monitor automatically, instead of just
        # enforcing that people have called push_monitor
        assert self.monitor.get_examples_seen() == 0
        self.monitor._sanity_check()




        X = model.get_input_space().make_theano_batch(name="%s[X]" % self.__class__.__name__)
        self.topo = not X.ndim == 2

        if config.compute_test_value == 'raise':
            if self.topo:
                X.tag.test_value = dataset.get_batch_topo(self.batch_size)
            else:
                X.tag.test_value = dataset.get_batch_design(self.batch_size)

        Y = T.tensor3(name="%s[Y]" % self.__class__.__name__)


        if self.cost.supervised:
            if config.compute_test_value == 'raise':
                _, Y.tag.test_value = dataset.get_batch_design(self.batch_size, True)

            self.supervised = True
            cost_value = self.cost(model, X, Y)

        else:
            self.supervised = False
            cost_value = self.cost(model, X)
        if cost_value is not None and cost_value.name is None:
            if self.supervised:
                cost_value.name = 'objective(' + X.name + ', ' + Y.name + ')'
            else:
                cost_value.name = 'objective(' + X.name + ')'

        if self.monitoring_dataset is not None:
            self.monitor.setup(dataset=self.monitoring_dataset,
                    cost=self.cost, batch_size=self.batch_size, num_batches=self.monitoring_batches,
                    extra_costs=self.monitoring_costs
                    )
            if self.supervised:
                ipt = (X, Y)
            else:
                ipt = X

            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            #LR is meaningless for ADADELTA, this will be the adaptive coefficient (which serves the same role) => printing var.
                        #if self.momentum:
             #   self.monitor.add_channel(name='momentum', ipt=ipt,
              #          val=self.momentum, dataset=monitoring_dataset, data_specs=(CompositeSpace([model.get_input_space(), model.get_output_space()]), ('features', 'targets')))
            '''
            Ypred = model.fprop(X)
            Y_ = (T.arange(0,96).dimshuffle('x','x',0)*Ypred).sum(axis = 2)
            y = monitoring_dataset.y
            the_y = T.matrix('targetsss')
            mse = Print('MSE')(T.mean(T.square(Y_-the_y)))
            funct = function(inputs=[X], outputs=mse)
            real_funct = function(inputs=[X,the_y], outputs=funct(), givens=[y=monitoring_dataset.y])
            self.monitor.add_channel(name='MSE', ipt=(y, X), val = 2, dataset=monitoring_dataset, prereqs=(funct))
            '''

        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i

        if self.cost.supervised:
            grads, updates = self.cost.get_gradients(model, X, Y)
        else:
            grads, updates = self.cost.get_gradients(model, X)

        for param in grads:
            assert param in params
        for param in params:
            assert param in grads

        for param in grads:
            if grads[param].name is None and cost_value is not None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})

        self.grads = grads

        #lr_scalers = model.get_lr_scalers()

        #for key in lr_scalers:
         #   if key not in params:
          #      raise ValueError("Tried to scale the learning rate on " +\
           #             str(key)+" which is not an optimization parameter.")

        #log.info('Parameter and initial learning rate summary:')
        #for param in params:
        #    param_name = param.name
        #    if param_name is None:
        #        param_name = 'anon_param'
            #lr = self.learning_rate.get_value()
            #log.info('\t' + param_name + ': ' + str(lr))

        #ADADELTA
        update_dict = {}
        idx = 0
        for g in grads:
            if g not in self.E_g2:
                self.E_g2[g] = T.pow(grads[g], 2)
            if g not in self.E_dx2:
                self.E_dx2[g] = theano.shared(numpy.cast['float32'](0))
            if g not in self.learning_rate:
                self.learning_rate[g] = theano.shared(value=numpy.cast['float32'](1), name='learning_rate')
                self.monitor.add_channel(name='learning_rate_EMA_' + str(g) + str(idx), ipt=ipt,
                    val=self.learning_rate[g], dataset=monitoring_dataset, data_specs = (CompositeSpace([model.get_input_space(), model.get_output_space()]), ('features', 'targets')))
                idx += 1


            self.E_g2[g] = self.decay_factor * self.E_g2[g] + (1 - self.decay_factor) * T.pow(grads[g], 2)
            up_g2 = self.E_g2[g] + numpy.cast[theano.config.floatX](1e-10)
            up_dx2 = self.E_dx2[g] + numpy.cast[theano.config.floatX](1e-10)
            lr = (T.sqrt(up_dx2)/T.sqrt(up_g2))
            up_lr = T.mean(numpy.cast[theano.config.floatX](0.95)*self.learning_rate[g] + numpy.cast[theano.config.floatX](1-0.95)*lr)
            update_dict[self.learning_rate[g]] = up_lr
            dx = -lr * grads[g]
            self.E_dx2[g] = numpy.cast[theano.config.floatX](self.decay_factor) * self.E_dx2[g] + numpy.cast[theano.config.floatX](1 - self.decay_factor) * T.pow(dx, 2)

            update_dict[g] = g + dx
        updates.update(update_dict)

        for param in params:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'
        model.censor_updates(updates)
        for param in params:
            update = updates[param]
            if update.name is None:
                update.name = 'censor(sgd_update(' + param.name + '))'
            for update_val in get_debug_values(update):
                if np.any(np.isinf(update_val)):
                    raise ValueError("debug value of %s contains infs" % update.name)
                if np.any(np.isnan(update_val)):
                    raise ValueError("debug value of %s contains nans" % update.name)


        with log_timing(log, 'Compiling sgd_update'):
            if self.supervised:
                fn_inputs = [X, Y]
            else:
                fn_inputs = [X]
            self.sgd_update = function(fn_inputs, updates=updates,
                                       name='sgd_update',
                                       on_unused_input='ignore',
                                       mode=self.theano_function_mode)
        self.params = params

    def train(self, dataset):
        #import pdb
        #import theano
        #pdb.set_trace()
        #theano.printing.debug_print()
        #theano.printing.pydotprint()

        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")
        model = self.model
        batch_size = self.batch_size

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            #this is sometimes very slow. we could get a huge speedup if we could
            #avoid having to run this everytime.
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None

        iterator = dataset.iterator(mode=self.train_iteration_mode,
                batch_size=self.batch_size, targets=self.supervised,
                topo=self.topo, rng = rng, num_batches = self.batches_per_iter)
        if self.topo:
            batch_idx = dataset.get_topo_batch_axis()
        else:
            batch_idx = 0
        if self.supervised:
            ind = 0
            for (batch_in, batch_target) in iterator:
                self.sgd_update(batch_in, batch_target)
                actual_batch_size = batch_in.shape[batch_idx]
                self.monitor.report_batch(actual_batch_size)
                for callback in self.update_callbacks:
                    callback(self)
        else:
            for batch in iterator:
                self.sgd_update(batch)
                actual_batch_size = batch.shape[0] # iterator might return a smaller batch if dataset size
                                                   # isn't divisible by batch_size
                self.monitor.report_batch(actual_batch_size)
                for callback in self.update_callbacks:
                    callback(self)

        # Make sure none of the parameters have bad values
        #this part is also sometimes very slow. Here again, if we can find a way
        #to speed it up, the gain could be significant.
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

    def continue_learning(self, model):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion.continue_learning(self.model)
