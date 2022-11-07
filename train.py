"""
Actually our own file.
"""

import numpy as np
from time import time
from keras import callbacks
from timeit import default_timer as timer
from dataset import get_dataset_reshaped, split_dataset, get_resphaped_dataset_paper, get_cb513, is_filtered
import model

data = np.load('dataset/cullpdb+profile_6133_filtered.npy')
data = data.reshape(5534, 700, 57)
print(data.shape)


def alpha_helix(matrix):
    for i in range(len(matrix)):

        if i % 1000 == 0:
            print(i, '/ ', len(matrix))

        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                if k < 22:
                    matrix[i][j][k] = data[i][j][k]
                elif k == 22:
                    if (data[i][j][27] == 1):
                        matrix[i][j][22] = 1
                    else:
                        matrix[i][j][22] = 0
                else:
                    matrix[i][j][k] = data[i][j][k+7]

    return matrix


newData = np.zeros([5534, 700, 50])
data_alpha = alpha_helix(newData)
print(data_alpha.shape)