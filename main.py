"""
The training script.
"""

import model
import dataset

should_consider_all_helixes = True

if should_consider_all_helixes:
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_helix_dataset(seed=100)
else:
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_alpha_helix_dataset(seed=100)


net = model.CNN_model()

history = net.fit(X_train, y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, y_val), callbacks=[model.checkpoint])

net.save('best_models/helix.h5')

