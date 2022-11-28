"""
Utils file with helper methods.
"""

import numpy as np

sequence_len = 700
total_features = 57
amino_acid_residues = 21
num_classes_orig = 8
cnn_width = 17


def get_dataset(path="dataset/cullpdb+profile_6133.npy"):
    dataset = np.load(path)
    dataset = np.reshape(dataset, (dataset.shape[0], sequence_len, total_features))
    ret = np.zeros((dataset.shape[0], dataset.shape[1], amino_acid_residues + num_classes_orig))
    ret[:, :, :amino_acid_residues] = dataset[:, :, 35:56]
    ret[:, :, amino_acid_residues:] = dataset[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + num_classes_orig]
    return ret


def split_dataset(dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    train_split = int(dataset.shape[0]*0.8)
    test_val_split = int(dataset.shape[0]*0.1)
    train = dataset[:train_split, :, :]
    test = dataset[train_split:train_split+test_val_split, :, :]
    validation = dataset[train_split+test_val_split:, :, :]

    return train, test, validation


def reshape_data(X, y):
    # Reshape X
    padding = np.zeros((X.shape[0], X.shape[2], int(cnn_width/2)))
    X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
    X = np.swapaxes(X, 1, 2)
    res = np.zeros((X.shape[0], X.shape[1] - cnn_width + 1, cnn_width, amino_acid_residues))
    for i in range(X.shape[1] - cnn_width + 1):
        res[:, i, :, :] = X[:, i:i+cnn_width, :]
    res = np.reshape(res, (X.shape[0]*(X.shape[1] - cnn_width + 1), cnn_width, amino_acid_residues))
    X = res[np.count_nonzero(res, axis=(1, 2))>(int(cnn_width/2)*amino_acid_residues), :, :]

    # Reshape y
    y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2]))
    y = y[~np.all(y == 0, axis=1)]

    return X, y



def get_helix_labels(labels):

    relevant_labels = labels[:, 3:6]
    reshaped_labels = np.zeros(relevant_labels.shape[0])
    reshaped_labels = np.any(relevant_labels, axis=1, out=reshaped_labels)

    return reshaped_labels


def get_alpha_helix_labels(labels):
    return labels[:, 5]


def get_helix_dataset(seed=100):
    dataset = get_dataset('dataset/cullpdb+profile_6133_filtered.npy')
    train, test, validation = split_dataset(dataset, seed)

    X_train, y_train = train[:, :, :amino_acid_residues], train[:, :, amino_acid_residues:]
    X_test, y_test = test[:, :, :amino_acid_residues], test[:, :, amino_acid_residues:]
    X_val, y_val = validation[:, :, :amino_acid_residues], validation[:, :, amino_acid_residues:]

    # Reshape data using the window width
    X_train, y_train = reshape_data(X_train, y_train)
    X_test, y_test = reshape_data(X_test, y_test)
    X_val, y_val = reshape_data(X_val, y_val)

    y_train = get_helix_labels(y_train)
    y_test = get_helix_labels(y_test)
    y_val = get_helix_labels(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_alpha_helix_dataset(seed=100):
    dataset = get_dataset('dataset/cullpdb+profile_6133_filtered.npy')
    train, test, validation = split_dataset(dataset, seed)

    X_train, y_train = train[:, :, :amino_acid_residues], train[:, :, amino_acid_residues:]
    X_test, y_test = test[:, :, :amino_acid_residues], test[:, :, amino_acid_residues:]
    X_val, y_val = validation[:, :, :amino_acid_residues], validation[:, :, amino_acid_residues:]

    # Reshape data using the window width
    X_train, y_train = reshape_data(X_train, y_train)
    X_test, y_test = reshape_data(X_test, y_test)
    X_val, y_val = reshape_data(X_val, y_val)

    y_train = get_alpha_helix_labels(y_train)
    y_test = get_alpha_helix_labels(y_test)
    y_val = get_alpha_helix_labels(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val