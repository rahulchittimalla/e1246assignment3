
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics


sentences = []
with open('ner.txt', encoding='latin1') as f:
    sentence = []
    for line in f:
        if line == "\n":
            sentences.append(sentence)
            sentence = []
        else:
            pair = line.strip("\n").split()
            sentence.append(tuple(pair))


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        
    def get_next(self):
        try:
            s = self.data[self.n_sent]
            self.n_sent += 1
            print(s)
            return s
        except:
            return None
        
getter = SentenceGetter(sentences)
sent = getter.get_next()
print(sent)

def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.label()': sent[i-1][1]
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.label()': sent[i+1][1]
        })
    else:
        features['EOS'] = True
    
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]


from sklearn_crfsuite import CRF
import scipy

crf = CRF(algorithm='lbfgs',
          c1=2,
          c2=0.2,
          max_iterations=100,
          all_possible_transitions=False)

crf.fit(X[:int(0.7*len(X))], y[:int(0.7*len(X))])

labels = list(crf.classes_)
labels

y_pred = crf.predict(X[int(0.8*len(X)):])
metrics.flat_f1_score(y[int(0.8*len(X)):], y_pred,
                      average='weighted', labels=labels)
print(metrics.flat_classification_report(
    y[int(0.8*len(X)):], y_pred, digits=3
))

import eli5
eli5.show_weights(crf, top=30)

