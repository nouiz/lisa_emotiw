from sgd import SGD
import numpy as np
from pylearn2.monitor import Monitor
from theano import config, shared
import theano.tensor as T
from theano import function
from theano.gof.op import get_debug_values
from theano.printing import Print
import logging
log = logging.getLogger(__name__)
from pylearn2.utils import sharedX
from pylearn2.utils.timing import log_timing


class KeypointSGD(SGD):
    """
    Stochastic Gradient Descent
    WRITEME: what is a good reference to read about this algorithm?

    A TrainingAlgorithm that does gradient descent on minibatches.

    """
    def setup(self, model, dataset):

        if self.cost is None:
            self.cost = model.get_default_cost()

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

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
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
            self.monitor.add_channel(name='learning_rate', ipt=ipt,
                    val=learning_rate, dataset=monitoring_dataset)
            if self.momentum:
                self.monitor.add_channel(name='momentum', ipt=ipt,
                        val=self.momentum, dataset=monitoring_dataset)
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

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " +\
                        str(key)+" which is not an optimization parameter.")

        log.info('Parameter and initial learning rate summary:')
        for param in params:
            param_name = param.name
            if param_name is None:
                param_name = 'anon_param'
            lr = learning_rate.get_value() * lr_scalers.get(param,1.)
            log.info('\t' + param_name + ': ' + str(lr))

        if self.momentum is None:
            updates.update( dict(safe_zip(params, [param - learning_rate * \
                lr_scalers.get(param, 1.) * grads[param]
                                    for param in params])))
        else:
            for param in params:
                inc = sharedX(param.get_value() * 0.)
                if param.name is not None:
                    inc.name = 'inc_'+param.name
                updated_inc = self.momentum * inc - learning_rate * lr_scalers.get(param, 1.) * grads[param]
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


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
