"""
Actually our own file.
"""

import numpy as np
from time import time
from keras import callbacks
from timeit import default_timer as timer
from dataset import get_dataset_reshaped, split_dataset, get_resphaped_dataset_paper, get_cb513, is_filtered
import model

import utils

dataset = utils.get_alpha_helix_data()
X_train, y_train, X_test, y_test, X_val, y_val = utils.split_dataset(dataset)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_val.shape)
# print(y_val.shape)

net = model.CNN_model()

start_time = timer()

history = net.fit(X_train, y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, y_val), callbacks=[model.checkpoint])

end_time = timer()
