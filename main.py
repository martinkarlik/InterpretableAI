
import keras
import model
import dataset
from lime.lime_text import LimeTextExplainer
from model_wrapper import ModelWrapper

X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_dataset_reshaped(seed=100)

net = model.CNN_model()
net.load_weights('best_models/draft_model.h5')
net_wrapper = ModelWrapper(net)

# Utils

AA_LABELS = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', '-']
FOLD_LABELS = ['E', 'B', 'H', 'G', 'I', 'T', 'S', '_']

ohe_text_map = {}


def one_hot_decoder(item, x=True):
    one_hot_labels = AA_LABELS if x else FOLD_LABELS
    sequence = ''
    for aa in item:
        max_label = (0, 0)
        for i, val in enumerate(aa):
            if val > max_label[0]:
                max_label = (val, i)
        if max_label[0] != 0:
            sequence += one_hot_labels[max_label[1]]
    ohe_text_map[sequence] = item
    return sequence


# def tokenize_aa_data(dataset, vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')):
#     separated_dataset = []
#     for item in dataset:
#         separated_string = ' '.join(item)
#         separated_string = re.sub(r'(\\ .)', r"\\\1", separated_string)
#         separated_dataset.append(separated_string)
#     return vectorizer.fit_transform(separated_dataset)
#
# vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')


def predict_text_to_ohe(item):
    #item_ohe = ohe_text_map[item[0]]
    #return net.predict(item_ohe)
    return net.predict(X_test)[idx]


X_train_text = [one_hot_decoder(x) for x in X_train]
X_test_text = [one_hot_decoder(x) for x in X_test]
# train_vectors = tokenize_aa_data(X_train_text, vectorizer)
# test_vectors = tokenize_aa_data(X_test_text, vectorizer)


# Some initial lime_text work (no data translation)

explainer = LimeTextExplainer(class_names=['Helix', 'Not a Helix'])

idx = 5
explanation = explainer.explain_instance(X_train_text[idx], net_wrapper.translate_and_predict, num_features=10, labels=y_train[idx])
print(explanation)


explainer = LimeTextExplainer(
    21,
    #X_train_text,
    #training_labels = y_train,
    #feature_names = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', '-'],
    class_names = ['Helix', 'Not helix'])

idx = 5
exp = explainer.explain_instance(X_test_text[idx], predict_text_to_ohe, num_features=10, labels=y_train)# labels=[0, 17])
