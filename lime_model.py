import re
import sys

import sklearn
from numpy import shape

import dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


AA_LABELS = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', '-']
FOLD_LABELS = ['E', 'B', 'H', 'G', 'I', 'T', 'S', '_']


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
    return sequence

# The data we want to tokenize is composed of a single sequence of characters so we need to prepare it to be tokenized
# This function adds spaces between each character as each aminoacid should be its own token.
# Control characters (like \n) will be reconstructed after their separation
# A custom token pattern is used since the default one does not accept character words

def tokenize_aa_data(dataset, vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')):
    separated_dataset = []
    for item in dataset:
        separated_string = ' '.join(item)
        separated_string = re.sub(r'(\\ .)', r"\\\1", separated_string)
        separated_dataset.append(separated_string)
    return vectorizer.fit_transform(separated_dataset)


# categories = ['G', 'soc.religion.christian'] # Helix/Not Helix
#newsgroups_train = fetch_20newsgroups(subset='train')  # Get training set for only select categories
#print(newsgroups_train.target)
#sys.exit()
# newsgroups_test = fetch_20newsgroups(subset='test', categories=categories) # Get test set for only select categories
# class_names = ['atheism', 'christian'] # Helix/Not Helix
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')
loadedDataset = dataset.get_dataset()
X_tr, X_v, X_te = dataset.split_like_paper(loadedDataset)
X_train, Y_train = dataset.get_data_labels(X_tr)
X_test, Y_test = dataset.get_data_labels(X_tr)
# train_vectors = tokenize_aa_data(X_test.data)
X_train_text = [one_hot_decoder(x) for x in X_train]
Y_train_text = [one_hot_decoder(y, x=False) for y in Y_train]
X_test_text = [one_hot_decoder(x) for x in X_test]
Y_test_text = [one_hot_decoder(y, x=False) for y in Y_test]
train_vectors = tokenize_aa_data(X_train_text, vectorizer)
test_vectors = tokenize_aa_data(X_test_text, vectorizer)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] #TODO temp

nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, Y_train_text)
pred = nb.predict(test_vectors)
# sklearn.metrics.f1_score(X_test.target, pred, average='weighted')
sklearn.metrics.f1_score(Y_test_text, pred, average='weighted')

c = make_pipeline(vectorizer, nb)


explainer = LimeTextExplainer(class_names=class_names)

idx = 5
exp = explainer.explain_instance(X_test_text[idx], c.predict_proba, labels=[0, 17])
print('Document id: %d' % idx)
tmp = nb.predict(X_test[idx])
print(tmp)
print('Predicted class =', tmp[0])
print('True class: ', one_hot_decoder(Y_test[idx], x=False))

print('Explanation for class %s' % class_names[0])
print('\n'.join(map(str, exp.as_list(label=0))))

exp = explainer.explain_instance(X_test_text[idx], c.predict_proba, num_features=6, top_labels=2)
print(exp.available_labels())
exp.show_in_notebook(text=False)

exp.show_in_notebook(text=X_test_text[idx], labels=(0,))
