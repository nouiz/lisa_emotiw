import sys
import cPickle
import os
import re
import time

import PIL.Image
from sklearn import preprocessing
import numpy
import theano
import ml
from jobman import DD
import jobman, jobman.sql
from utils import tile_raster_images


def save_weights(model, epoch):
    if epoch % 5 != 0:
        return
    tile_j = int(numpy.ceil(numpy.sqrt(model.n_hiddens)))
    tile_i = int(numpy.ceil(float(model.n_hiddens) / tile_j))

    W = (model.W.T)

    image = PIL.Image.fromarray(tile_raster_images(
             X = W,
             img_shape = (31, 39), tile_shape = (tile_j, tile_i),
             tile_spacing=(1,1)))
    image.save('weights_%d.png' % epoch)


def main(n_hiddens=400,
         n_layers=2,
		 learning_rate=0.001,
		 momentum=0.5,
         use_nesterov = False,
		 rbm_learning_rate=[0.001, 0.001],
		 rbm_epochs=32,
         mean_pooling=False,
		 rbm_batch_size=100,
		 example_dropout=32,
         seed=19314312,
         hidden_dropout=.5,
         enable_standardization=False,
         center_grads=False,
         rmsprop = True,
         max_col_norm = 1.8356,
		 l2=None,
		 train_epochs=240,
		 K=1,
         loss_based_pooling=False,
         rho=0.96,
		 features="minimal.pca",
		 state=None,
		 channel=None,
		 **kwargs):

    print "Hypeparams:"
    import inspect
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        print "    %s = %s" % (i, values[i])

    print "Loading dataset..."

    numpy.random.seed(0x7265257d5f)

    LABELS = ["Disgust",  "Fear",  "Happy",  "Neutral",  "Sad",  "Surprise", "Angry"]

    train_x = []
    train_y = []
    print "Loading training set..."
    for directory, dirnames, filenames in os.walk("/data/lisa/data/faces/EmotiWTest/audio_features_mixed/Train"):
        for filename in filenames:
            if filename.find("%s.pkl" % features) != -1:
                feat = numpy.load(os.path.join(directory, filename))
                targ = numpy.argmax(map(lambda x: directory.find(x) != -1, LABELS))

                train_x.append(numpy.asarray(feat, theano.config.floatX))
                train_y.append(targ)

    train_y = numpy.asarray(train_y)

    pretrain_x = numpy.load("/data/lisatmp/dauphiya/emotiw/mlp_audio/train_x_%s.npy" % features)

    valid_x = []
    valid_y = []
    print "Loading validation set..."
    for directory, dirnames, filenames in os.walk("/data/lisa/data/faces/EmotiW/complete_audio_features/Val"):
        for filename in filenames:
            if filename.find("%s.pkl" % features) != -1:
                feat = numpy.load(os.path.join(directory, filename))
                targ = numpy.argmax(map(lambda x: directory.find(x) != -1, LABELS))

                valid_x.append(numpy.asarray(feat, theano.config.floatX))
                valid_y.append(targ)

    valid_y = numpy.asarray(valid_y, dtype=theano.config.floatX)

    test_x = []
    test_y = []
    print "Loading test set..."

    for directory, dirnames, filenames in os.walk("/data/lisa/data/faces/EmotiWTest/audios_features/Test"):
        for filename in filenames:
            if filename.find("%s.pkl" % features) != -1:
                feat = numpy.load(os.path.join(directory, filename))
                targ = numpy.argmax(map(lambda x: directory.find(x) != -1, LABELS))

                test_x.append(numpy.asarray(feat, theano.config.floatX))
                test_y.append(targ)

    test_y = numpy.asarray(test_y, dtype=theano.config.floatX)

    means = numpy.asarray(numpy.sum([x.sum(0) for x in train_x], 0) / sum([x.shape[0] for x in train_x]), dtype=theano.config.floatX)
    #valid_means = means
    #test_means = means
    valid_means = numpy.asarray(numpy.sum([x.sum(0) for x in valid_x], 0) / sum([x.shape[0] for x in valid_x]), dtype=theano.config.floatX)
    test_means = numpy.asarray(numpy.sum([x.sum(0) for x in test_x], 0) / sum([x.shape[0] for x in test_x]), dtype=theano.config.floatX)

    train_inds = range(len(train_x))
    numpy.random.shuffle(train_inds)

    print "Building model..."

    layers = n_layers * [('L', n_hiddens)] +  [('S', train_y.max() + 1)]

    if type(rbm_learning_rate) in [str, unicode]:
        rbm_learning_rate = eval(rbm_learning_rate)

    if type(rbm_learning_rate) is float:
        rbm_learning_rate = len(layers) * [rbm_learning_rate]

    model = ml.MLP(n_in=train_x[0].shape[1],
                   layers=layers,
                   learning_rate=learning_rate,
                   l2=l2,
                   rho=rho,
                   rmsprop=rmsprop,
                   center_grads=center_grads,
                   max_col_norm=max_col_norm,
                   loss_based_pooling=loss_based_pooling,
                   hidden_dropout=hidden_dropout,
                   use_nesterov=use_nesterov,
                   mean_pooling=mean_pooling,
                   enable_standardization=enable_standardization,
                   momentum=momentum)

    rbms = []
    def project(x):
       for rbm in rbms:
           x = rbm.mean_h(x)
       return x

    for i in range(len(layers) - 1):
       print "Training RBM %d..." % i
       if i == 0:
           rbm = ml.GaussianRBM(n_hiddens=n_hiddens,
                                epsilon=rbm_learning_rate[i],
                                n_samples=rbm_batch_size,
                                epochs=rbm_epochs,
                                K=K)
       else:
            rbm = ml.BinaryRBM(n_hiddens=n_hiddens,
                                 epsilon=rbm_learning_rate[i],
                                 n_samples=rbm_batch_size,
                                 epochs=rbm_epochs,
                                 K=K)

       rbm.fit(pretrain_x, project=project, verbose=True)

       model.layers[i].W.set_value(rbm.W)
       model.layers[i].b.set_value(rbm.c)

       numpy.save("rbm_W_%d.npy" % i, rbm.W)

       rbms.append(rbm)

    print "Training model..."

    epoch_times = []

    best_train_error = float('inf')

    best_valid_error = float('inf')

    for epoch in range(train_epochs):
        begin = time.time()

        losses = []

        for minibatch in range(len(train_inds)):
            x = train_x[train_inds[minibatch]] - means
            y = train_y[[train_inds[minibatch]]]

            inds = range(x.shape[0])
            numpy.random.shuffle(inds)
            inds = inds[:example_dropout]

            losses.append(model.train(x[inds], y))
        end = time.time()

        loss = numpy.mean(losses)

        train_error = 0.
        for minibatch in range(len(train_inds)):
            x = train_x[train_inds[minibatch]] - means
            y = train_y[[train_inds[minibatch]]]

            train_error += (y != model.output(x))[0]
        train_error /= len(train_inds)

        valid_error = 0.
        for minibatch in range(len(valid_y)):
            x = valid_x[minibatch] - valid_means
            y = valid_y[[minibatch]]

            valid_error += (y != model.output(x))[0]
        valid_error /= len(valid_y)

        epoch_times.append((end - begin) / 60)

        mean_epoch_time = numpy.mean(epoch_times)

        if train_error < best_train_error:
            best_train_error = train_error
        elif epoch > 50:
            model.trainer.learning_rate.set_value(numpy.asarray(0.99 * model.trainer.learning_rate.get_value(), dtype=theano.config.floatX))

        if valid_error < best_valid_error:
            best_valid_error = valid_error

            model.save()

        elif epoch > 50:
            model.trainer.learning_rate.set_value(numpy.asarray(0.99 * model.trainer.learning_rate.get_value(), dtype=theano.config.floatX))

        print "epoch = %d, mean_time = %.2f, loss = %.4f, train_error = %.4f, valid_error = %.4f, learning rate = %.4f" % (epoch, mean_epoch_time, loss, train_error, valid_error, model.trainer.learning_rate.get_value())
        print "best validation error %.4f" % (best_valid_error)
        if channel != None:
            state.epoch = epoch
            state.epochtime = mean_epoch_time
            state.loss = loss
            state.trainerror = best_train_error
            state.validerror = best_valid_error

            channel.save()


def jobman_entrypoint(state, channel):
    main(state=state, channel=channel, **state)
    return channel.COMPLETE


def jobman_insert_random(n_jobs, table_name="emotiw_mlp_audio_sigm_fixed_pool2_mixed_norbm3"):

    JOBDB = 'postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=' + table_name
    EXPERIMENT_PATH = "experiment_cg_2layer_sigm_hyper2_fixed2_pool2_save_mixed_norbm3.jobman_entrypoint"
    nlr = 45
    learning_rates = numpy.logspace(numpy.log10(0.0008), numpy.log10(0.1), nlr)
    max_col_norms = [1.8256, 1.5679, 1.2124, 0.98791]
    rhos = [0.96, 0.92, 0.88]
    jobs = []

    for _ in range(n_jobs):

        job = DD()
        id_lr = numpy.random.random_integers(0, nlr-1)
        rnd_maxcn = numpy.random.random_integers(0, len(max_col_norms)-1)
        rnd_rho = numpy.random.random_integers(0, len(rhos)-1)
        job.n_hiddens = numpy.random.randint(80, 500)
        job.n_layers = numpy.random.random_integers(1, 2)
        job.learning_rate = learning_rates[id_lr]
        job.momentum = 10.**numpy.random.uniform(-1, -0)
        job.rmsprop = 1
        job.rho = rhos[rnd_rho]
        job.validerror = 0.0
        job.loss = 0.0
        job.seed = 1938471
        job.rbm_epochs = 0
        job.epoch = 0
        job.epoch_time = 0
        job.use_nesterov = 1
        job.trainerror = 0.0
        job.features = "full.pca"
        job.max_col_norm = max_col_norms[rnd_maxcn]
        job.example_dropout = numpy.random.randint(60, 200)
        job.tag = "sigm_norm_const_fixed_pool2_norbm3"

        jobs.append(job)
        print job

    answer = raw_input("Submit %d jobs?[y/N] " % len(jobs))

    if answer == "y":
        numpy.random.shuffle(jobs)

        db = jobman.sql.db(JOBDB)
        for job in jobs:
            job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
            jobman.sql.insert_dict(job, db)

        print "inserted %d jobs" % len(jobs)
        print "To run: jobdispatch --condor --mem=3G --gpu --env=THEANO_FLAGS='floatX=float32, device=gpu' --repeat_jobs=%d jobman sql -n 1 '%s' ." % (len(jobs), JOBDB)


def produit_cartesien_jobs(val_dict):
    job_list = [DD()]
    all_keys = val_dict.keys()

    for key in all_keys:
        possible_values = val_dict[key]
        new_job_list = []
        for val in possible_values:
            for job in job_list:
                to_insert = job.copy()
                to_insert.update({key: val})
                new_job_list.append(to_insert)
        job_list = new_job_list

    return job_list


def jobman_insert():

    JOBDB = 'postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table='
    EXPERIMENT_PATH = "experiment_cg_2layer_hyper2_norbm.jobman_entrypoint"
    JOB_VALS = {
        'n_hiddens' : [1024],
        'n_layers' : [1],
        'learning_rate' : [0.01, 0.001],
        'rbm_learning_rate' : [0.001, 0.0001],
        'rbm_epochs' : [32,],
        'example_dropout' : [32],
        'l2' : [0],
        'train_epochs' : [200],
        'K' : [1, 5, 10],
        'tag' : ['pretrain-all2'],
        }

    o
    jobs = produit_cartesien_jobs(JOB_VALS)

    for job in jobs:
        print job

    answer = raw_input("Submit %d jobs?[y/N] " % len(jobs))
    if answer == "y":
        numpy.random.shuffle(jobs)

        db = jobman.sql.db(JOBDB)
        for job in jobs:
            job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
            jobman.sql.insert_dict(job, db)

        print "inserted %d jobs" % len(jobs)
        print "To run: jobdispatch --condor --mem=3G --gpu --env=THEANO_FLAGS='floatX=float32, device=gpu' --repeat_jobs=%d jobman sql -n 1 'postgres://dauphiya@opter.iro.umontreal.ca/dauphiya_db/emotiw_mlp_audio' ." % len(jobs)

def view(table="emotiw_mlp_audio_sigm_fixed_pool2_mixed_norbm3",
         tag="sigm_norm_const_fixed_pool2_mixed2",
         user="gulcehrc",
         password="",
         database="gulcehrc_db",
         host="opter.iro.umontreal.ca"):
    """
    View all the jobs in the database.
    """
    import commands
    import sqlalchemy
    import psycopg2

    # Update view
    url = "postgres://%s:%s@%s/%s/" % (user, password, host, database)
    commands.getoutput("jobman sqlview %s%s %s_view" % (url, table, table))

    # Display output
    def connect():
        return psycopg2.connect(user=user, password=password,
                                database=database, host=host)

    engine = sqlalchemy.create_engine('postgres://', creator=connect)
    conn = engine.connect()
    experiments = sqlalchemy.Table('%s_view' % table,
                                   sqlalchemy.MetaData(engine), autoload=True)

    columns = [experiments.columns.id,
               experiments.columns.jobman_status,
               experiments.columns.rho,
               experiments.columns.nhiddens,
               experiments.columns.learningrate,
               experiments.columns.momentum,
               experiments.columns.exampledropout,
               experiments.columns.usenesterov,
               experiments.columns.epoch,
               experiments.columns.nlayers,
               experiments.columns.trainerror,
               experiments.columns.maxcolnorm,
               experiments.columns.validerror,]

    results = sqlalchemy.select(columns,
                                order_by=[experiments.columns.tag,
                                    sqlalchemy.desc(experiments.columns.validerror)]).execute()

    results = [map(lambda x: x.name, columns)] + list(results)

    def get_max_width(table, index):
        """Get the maximum width of the given column index"""
        return max([len(format_num(row[index])) for row in table])

    def format_num(num):
        """Format a number according to given places.
        Adds commas, etc. Will truncate floats into ints!"""
        try:
            if "." in num:
                return "%.7f" % float(num)
            else:
                return int(num)
        except (ValueError, TypeError):
            return str(num)

    col_paddings = []

    for i in range(len(results[0])):
        col_paddings.append(get_max_width(results, i))

    for row_num, row in enumerate(results):
        for i in range(len(row)):
            col = format_num(row[i]).ljust(col_paddings[i] + 2) + "|"
            print col,
        print

        if row_num == 0:
            for i in range(len(row)):
                print "".ljust(col_paddings[i] + 1, "-") + " +",
            print

if __name__ == "__main__":
    if "insert" in sys.argv:
        jobman_insert_random(int(sys.argv[2]))
    elif "view" in sys.argv:
        view()
    elif "graph" in sys.argv:
        graph()
    else:
        main(n_hiddens=193,
            learning_rate=0.00124084681414,
            momentum=0.259071749315,
            features="full.pca",
            example_dropout=148,
            max_col_norm=0.98791,
            nlayers=2,
            rho=0.96,
            train_epochs=400,
            #seed=838609,
            #center_grads=False,
            #loss_based_pooling=False,
            #enable_standardization=False,
            use_nesterov=True)

