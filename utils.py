"""
Utils file with helper methods.
"""

import numpy as np


def get_alpha_helix_data():

    data = np.load('dataset/cullpdb+profile_6133_filtered.npy')
    data = data.reshape(5534, 700, 57)

    transormed_data = np.zeros([5534, 700, 50])

    for i in range(len(transormed_data)):

        if i % 1000 == 0:
            print(i, '/ ', len(transormed_data))

        for j in range(len(transormed_data[i])):
            for k in range(len(transormed_data[i][j])):
                if k < 22:
                    transormed_data[i][j][k] = data[i][j][k]
                elif k == 22:
                    if data[i][j][27] == 1:
                        transormed_data[i][j][22] = 1
                    else:
                        transormed_data[i][j][22] = 0
                else:
                    transormed_data[i][j][k] = data[i][j][k+7]

    return transormed_data


def split_data():
    pass