"""A generic module to read data."""
import numpy as np
import collections
from tensorflow.python.framework import dtypes


class DataSet(object):
    """Dataset class object."""

    def __init__(self, features, labels,
            fake_data=False, dtype=dtypes.float64):
        """Initialize the class."""

        self._features         = features
        self._num_examples     = features.shape[0]
        self._labels           = labels
        self._epochs_completed = 0
        self._index_in_epoch   = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels   = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._features[start:end], self._labels[start:end]


def read_data_sets(data_dir, fake_data=False, dtype=dtypes.float64):
    """Set the features and labels."""

    # TODO if npy not exist, loadtxt and export to npy
    all_features = np.load(data_dir + 'pubmed.feature.npy')
    all_labels   = np.load(data_dir + 'pubmed.label.npy')

    tot = all_features.shape[0]
    assert tot == all_labels.shape[0], "train features' number of items doesn't match with labels'"

    num_training   = tot // 2
    num_validation = tot // 4
    num_test       = tot - num_training - num_validation

    mask = range(num_training)
    train_features = all_features[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_validation)
    validation_features = all_features[mask]
    validation_labels = all_labels[mask]

    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    test_features = all_features[mask]
    test_labels = all_labels[mask]

    train      = DataSet(train_features,      train_labels,      dtype=dtype)
    validation = DataSet(validation_features, validation_labels, dtype=dtype)
    test       = DataSet(test_features,       test_labels,       dtype=dtype)

    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)

