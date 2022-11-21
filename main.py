
import keras
import model
import dataset
from lime.lime_text import LimeTextExplainer

# Some initial lime_text work (no data translation)

net = model.CNN_model()
net.load_weights('best_models/draft_model.h5')
print(net)

X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_dataset_reshaped(seed=100)
explainer = LimeTextExplainer(class_names=['1', '0'])

idx = 5
explanation = explainer.explain_instance("ABCDEFGHIJKLMNOPQ", net.predict, num_features=10, labels=(1,))
print(explanation)