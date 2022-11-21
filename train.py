"""
The training script.
"""

import numpy as np
from time import time
from keras import callbacks
from timeit import default_timer as timer
import model
import dataset

X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_dataset_reshaped(seed=100)

print(X_test.shape)

# net = model.CNN_model()
#
# history = net.fit(X_train, y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
#                         validation_data=(X_val, y_val), callbacks=[model.checkpoint])
#
# net.save('best_models/draft_model.h5')

