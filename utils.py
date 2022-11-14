"""
Utils file with helper methods.
"""

sequence_len = 700
total_features = 57
amino_acid_residues = 21
num_classes = 8

cnn_width = 17

import numpy as np


def get_alpha_helix_dataset(path=None):

    data = np.load(path)
    data = data.reshape(5534, 700, 57)

    transormed_data = np.zeros([5534, 700, 22])
    transormed_data[:, :, :21] = data[:, :, 35:56]
    transormed_data[:, :, 21] = data[:, :, 27]

    return transormed_data


def split_dataset(dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    train_split = int(dataset.shape[0]*0.8)
    test_val_split = int(dataset.shape[0]*0.1)
    train = dataset[:train_split, :, :]
    test = dataset[train_split:train_split+test_val_split, :, :]
    validation = dataset[train_split+test_val_split:, :, :]

    return train, test, validation


def reshape_data(X):
    padding = np.zeros((X.shape[0], X.shape[2], int(cnn_width/2)))
    X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
    X = np.swapaxes(X, 1, 2)
    res = np.zeros((X.shape[0], X.shape[1] - cnn_width + 1, cnn_width, amino_acid_residues))
    for i in range(X.shape[1] - cnn_width + 1):
        res[:, i, :, :] = X[:, i:i+cnn_width, :]
    res = np.reshape(res, (X.shape[0]*(X.shape[1] - cnn_width + 1), cnn_width, amino_acid_residues))
    res = res[np.count_nonzero(res, axis=(1,2))>(int(cnn_width/2)*amino_acid_residues), :, :]
    return res


def reshape_labels(labels):
    y = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1))
    y = y[~np.all(y == 0, axis=1)]
    return y


def get_dataset_reshaped(seed=1000):
    dataset = get_alpha_helix_dataset('dataset/cullpdb+profile_6133_filtered.npy')
    train, test, validation = split_dataset(dataset, seed)

    X_train, y_train = train[:, :, :21], train[:, :, 21]
    X_test, y_test = test[:, :, :21], test[:, :, 21]
    X_val, y_val = validation[:, :, :21], validation[:, :, 21]

    print(X_train.shape)
    print(y_train.shape)
    # Reshape data using the window width
    X_train = reshape_data(X_train)
    X_test = reshape_data(X_test)
    X_val = reshape_data(X_val)

    y_train = reshape_labels(y_train)
    y_test = reshape_labels(y_test)
    y_val = reshape_labels(y_val)

    print(X_train.shape)
    print(y_train.shape)

    return X_train, y_train, X_test, y_test, X_val, y_val