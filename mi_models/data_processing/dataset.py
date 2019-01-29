# -*- coding: utf-8 -*-
# @time      : 2019/1/24 13:27
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : dataset.py

import numpy as np


def dense_to_one_hot(labels_dense, num_classes):
    '''
    Convert class labels from scalars to one-hot vectors.
    :param labels_dense:
    :param num_classes:
    :return:
    '''
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel().astype(int)] = 1
    return labels_one_hot


def split_train_val_dataset(train_X, train_Y, train_rate):
    # n_train = int(len(train_X) * train_rate)
    # n_val = int(len(train_X) * (1 - train_rate))
    n_train = int(len(train_X) * train_rate)
    train_idx = np.array(range(int(len(train_X) * train_rate)))
    np.random.shuffle(train_idx)
    val_idx = np.array(range(int(len(train_X) * (1 - train_rate))))
    np.random.shuffle(val_idx)
    # train_idx = np.random.shuffle(np.array(range(n_train)))
    # val_idx = np.random.shuffle(np.array(range(n_val)))
    return train_X[:n_train], train_Y[:n_train], train_X[n_train:], train_Y[n_train:]
    # return train_X[train_idx], train_Y[train_idx], train_X[val_idx], train_Y[val_idx]


class DataSet(object):
    def __init__(self, images, labels, acc_input):
        assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        self._images = images
        self._labels = labels
        self._acc_inputs = acc_input
        self._epochs_completed = 0
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        '''
        Return the next `batch_size` examples from this data set
        :param batch_size:
        :return:
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._acc_inputs[start:end]


if __name__ == '__main__':
    image = np.random.random(size=(10000, 10, 10))
    label = np.zeros(shape=(10000))
    d = DataSet(image, label)
    b = d.next_batch(30)
    import pprint

    pprint.pprint(b)
