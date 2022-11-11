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


data = utils.get_alpha_helix_data()

X_train, X_val, X_test, Y_train, Y_val, Y_test = utils.split_data()


net = model.CNN_model()

start_time = timer()

history = net.fit(X_train, Y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=[model.checkpoint])

end_time = timer()
