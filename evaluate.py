# MIT License
#
# Copyright (c) 2017 Luca Angioloni
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from time import time
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset_reshaped
import model

start_time = timer()

print("Collecting Dataset...")


X_train, X_val, X_test, Y_train, Y_val, Y_test = get_dataset_reshaped(seed=100)

end_time = timer()
print("\n\nTime elapsed getting Dataset: " + "{0:.2f}".format((end_time - start_time)) + " s")

net = model.CNN_model()

#load Weights
net.load_weights("best_models/CullPDB6133-best - 0.721522.hdf5")

scores = net.evaluate(X_test, Y_test)
print(scores)
print("Loss: " + str(scores[0]) + ", Accuracy: " + str(scores[1]) + ", MAE: " + str(scores[2]))

predictions = net.predict(X_test)
predictions_indices = np.argmax(predictions, axis=1)

count_correct_helix = 0
all_helixes = 0

for i in np.arange(len(predictions_indices)):

    is_helix = 2 <= np.argmax(Y_test[i]) <= 4
    did_predict_helix = 2 <= predictions_indices[i] <= 4

    if is_helix:
        all_helixes += 1

    if is_helix == did_predict_helix:
        count_correct_helix += 1

print("Accuracy of helix/no-helix classification: {:.2f}%".format(count_correct_helix / Y_test.shape[0] * 100))
print("Ratio of helixes in dataset: {:.2f}%".format(all_helixes / Y_test.shape[0] * 100))

