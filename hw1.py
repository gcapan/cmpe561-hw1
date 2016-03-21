# coding=utf-8
"""
From a parent directory of training/test directories of category directories of input documents,
learns a NB classifier from the training documents, classifies the test documents, and reports the accuracy
"""

import argparse
import os
import re

from preprocessor import tokenizer as prep
from classifier import nb

#Experiment- BoW
def experiment(train, test, text_normalizer, word_normalizer, word_filter,
               ngraml= None, pronouns = False, char_ngrams = False, char_ngraml = None, complexity = False, alpha = 1.0):

    train_docs = dict([(c, [prep.process(document, text_normalizer, word_normalizer, word_filter, ngraml = ngraml,
                                pronouns = pronouns, char_ngrams = char_ngrams, complexity = complexity)
                   for document in train[c]]) for c in train])

    test_docs = dict([(c, [prep.process(document, text_normalizer, word_normalizer, word_filter, ngraml = ngraml,
                                pronouns = pronouns, char_ngrams = char_ngrams, complexity = complexity)
                   for document in test[c]]) for c in test])

    nb_model = nb.build(train_docs)
    actual = []
    predicted = []
    for c in test_docs:
        for d in test_docs[c]:
            actual.append(c)
            predicted.append(nb.estimate(d, nb_model, alpha))

    return nb.evaluate(actual, predicted)

def report_experiment(ex, doc, format=None):
    print doc
    if format == None:
        for key in ['precision', 'recall', 'f1']:
            print ex[key]['micro-averaged'], ex[key]['macro-averaged']

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--train", help = "Path for the training documents")
parser.add_argument("--test", help = "Path for the test documents")

args = parser.parse_args()
train = dict([(c, os.listdir(os.path.join(args.train, c)))
         for c in os.listdir(args.train) if not re.match(r'^\.', c)])

for c in train:
    train[c] = [os.path.join(args.train, c, d) for d in train[c] if not re.match(r'^\.', d)]

test = dict([(c, os.listdir(os.path.join(args.test, c)))
         for c in os.listdir(args.test) if not re.match(r'^\.', c)])

for c in test:
    test[c] = [os.path.join(args.test, c, d) for d in test[c] if not re.match(r'^\.', d)]

text_normalizer = lambda s: s.lower().replace('"', '')
word_filter = lambda s: len(s) > 1 and re.search(r'\d', s) is None
word_normalizer = lambda w: w[:w.find(ur"’'")] if re.search(ur"’'", w) is not None else w


for alpha in [0.01, 0.1, 1]:
    #Experiment- Bow
    report_experiment(experiment(train, test, text_normalizer, word_normalizer, word_filter, alpha = alpha),
                      'BoW, smoothing: '+str(alpha))

    #Experiment- BoW + bigrams
    report_experiment(experiment(train, test, text_normalizer, word_normalizer, word_filter, ngraml = (2,3),
                                 alpha =alpha), doc = 'BoW + bigrams, smoothing: '+str(alpha))

    #Experiment- BoW + bigrams + pronouns
    #Note that pronouns are overcounted
    report_experiment(experiment(train, test, text_normalizer, word_normalizer, word_filter,
                             ngraml = (2,3), pronouns=True, alpha = alpha),
                      doc = 'BoW + bigrams + pronouns, smoothing: '+str(alpha))

    #Experiment- BoW + bigrams + pronouns + complexity
    report_experiment(experiment(train, test, text_normalizer, word_normalizer, word_filter, ngraml = (2,3),
           pronouns = True, char_ngrams = False, complexity = True, alpha = alpha),
                  'BoW + bigrams + pronouns + complexity features, smoothing: '+str(alpha))

