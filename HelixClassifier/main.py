"""
The training script.
"""

import model
import dataset

should_consider_all_helixes = False

if should_consider_all_helixes:
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_helix_dataset(seed=100)
else:
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_alpha_helix_dataset(seed=100)


net = model.CNN_model()

history = net.fit(X_train, y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, y_val), callbacks=[model.checkpoint, model.early_stop, model.logger])

# alpha_helix_window_5
net.save('best_models/alpha_helix_window_13.h5')



"""
Results (model: val_accuracy):
any_helix_window_17: 0.9105

alpha_helix_window_5: 0.8733
alpha_helix_window_13: 0.9204
alpha_helix_window_17: 0.9253

"""