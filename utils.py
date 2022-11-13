"""
Utils file with helper methods.
"""

import numpy as np


def get_alpha_helix_data():

    data = np.load('dataset/cullpdb+profile_6133_filtered.npy')
    data = data.reshape(5534, 700, 57)

    transormed_data = np.zeros([5534, 700, 23])
    transormed_data[:, :, :22] = data[:, :, :22]
    transormed_data[:, :, 22] = data[:, :, 27]

    # for i in range(len(transormed_data)):
    #
    #     if i % 1000 == 0:
    #         print(i, '/ ', len(transormed_data))
    #
    #     for j in range(len(transormed_data[i])):
    #         for k in range(len(transormed_data[i][j])):
    #             if k < 22:
    #                 transormed_data[i][j][k] = data[i][j][k]
    #             elif k == 22:
    #                 if data[i][j][27] == 1:
    #                     transormed_data[i][j][22] = 1
    #                 else:
    #                     transormed_data[i][j][22] = 0
    #             else:
    #                 transormed_data[i][j][k] = data[i][j][k+7]

    return transormed_data


def split_dataset(dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    train_split = int(dataset.shape[0]*0.8)
    test_val_split = int(dataset.shape[0]*0.1)
    train = dataset[:train_split, :, :]
    test = dataset[train_split:train_split+test_val_split, :, :]
    validation = dataset[train_split+test_val_split:, :, :]

    X_train = train[:, :, :22]
    y_train = train[:, :, 22]
    X_test = test[:, :, :22]
    y_test = test[:, :, 22]
    X_val = validation[:, :, :22]
    y_val = validation[:, :, 22]

    return X_train, y_train, X_test, y_test, X_val, y_val

