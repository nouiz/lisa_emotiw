from sklearn.decomposition import PCA
import numpy

def load_datasets(files=None, normalize=True):
    assert files is not None
    datasets = []
    for np_file in files:
        X = numpy.load(np_file)

        if normalize:
            X -= X.min(0)
            X /= X.max(0)
            X = 2.0 * X - 1.

        datasets.append(X)
    return datasets

def run_pca(training_datasets, validation_datasets, train_output, valid_output, center=True,
        nc=40):
    traininig_data = load_datasets(files=training_datasets)
    validation_data = load_datasets(files=validation_datasets)
    #test_datasets = load_datasets(files=test_datasets)

    ctraining_data = numpy.concatenate(traininig_data, axis=1)
    cvalidation_data = numpy.concatenate(validation_data, axis=1)
    #ctest_data = numpy.concatenate(test_datasets, axis=1)

    ctraining_data = ctraining_data - ctraining_data.mean()
    #ctest_data = ctraining_data - ctraining_data.mean()
    cvalidation_data = cvalidation_data - cvalidation_data.mean()

    pca_train = PCA(n_components=nc)
    pca_train.whiten = True
    pca_train.fit(ctraining_data)

    pca_valid = PCA(n_components=nc)
    pca_valid.whiten = True
    pca_valid.fit(cvalidation_data)

    numpy.save(train_output, pca_train.transform(ctraining_data))
    numpy.save(valid_output, pca_valid.transform(cvalidation_data))

if __name__=="__main__":
    training_datasets =\
    ["/data/lisatmp2/EmotiWTest/features_for_xavierb/afew2_train_ramanan1_emmanuel_features.npy",
            "/data/lisatmp2/EmotiWTest/audio_extracted_features/audio_mlp_train_feats.npy"]
    validation_datasets =\
    ["/data/lisatmp2/EmotiWTest/features_for_xavierb/afew2_valid_ramanan1_emmanuel_features.npy",
            "/data/lisatmp2/EmotiWTest/audio_extracted_features/audio_mlp_valid_feats.npy"]
    nc = 40
    valid_output = "valid_data_%d.npy" % nc
    train_output = "train_data_%d.npy" % nc

    run_pca(training_datasets, validation_datasets, train_output, valid_output, nc=nc)
