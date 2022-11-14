import re
import sys

import sklearn
from numpy import shape

import dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


# The data we want to tokenize is composed of a single sequence of characters so we need to prepare it to be tokenized
# This function adds spaces between each character as each aminoacid should be its own token.
# Control characters (like \n) will be reconstructed after their separation
# A custom token pattern is used since the default one does not accept character words


def tokenize_aa_data(dataset, vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True,                                                                 token_pattern=r'(?u)\b\w+\b')):
    separated_dataset = []
    for item in dataset:
        separated_string = ' '.join(item)
        separated_string = re.sub(r'(\\ .)', r"\\\1", separated_string)
        separated_dataset.append(separated_string)
    return vectorizer.fit_transform(separated_dataset)


# categories = ['G', 'soc.religion.christian'] # Helix/Not Helix
# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories) # Get training set for only select categories
# newsgroups_test = fetch_20newsgroups(subset='test', categories=categories) # Get test set for only select categories
# class_names = ['atheism', 'christian'] # Helix/Not Helix

X_train, X_val, X_test = dataset.split_like_paper(dataset.get_dataset())
class_names = ''.split('ABCDEFGH') #TODO temp
train_vectors = tokenize_aa_data(X_test.data)
test_vectors = tokenize_aa_data(X_test.data)

nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, X_train.target)
pred = nb.predict(test_vectors)
sklearn.metrics.f1_score(X_test.target, pred, average='weighted')


c = make_pipeline(vectorizer, nb)


explainer = LimeTextExplainer(class_names=class_names)

idx = 1340
exp = explainer.explain_instance(X_test.data[idx], c.predict_proba, num_features=6, labels=[0, 17])
print('Document id: %d' % idx)
print('Predicted class =', class_names[nb.predict(test_vectors[idx]).reshape(1, -1)[0, 0]])
print('True class: %s' % class_names[X_test.target[idx]])

print('Explanation for class %s' % class_names[0])
print('\n'.join(map(str, exp.as_list(label=0))))

exp = explainer.explain_instance(X_test.data[idx], c.predict_proba, num_features=6, top_labels=2)
print(exp.available_labels())
exp.show_in_notebook(text=False)

exp.show_in_notebook(text=X_test.data[idx], labels=(0,))
