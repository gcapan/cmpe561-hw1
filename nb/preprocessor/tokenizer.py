import os
from itertools import islice
from collections import Counter
import codecs

import re

SENTENCE_RE = r'[\s|^]+[^!?.]+[!?.]+\s'
WORD_SPLITBY_RE = r'[\s|?.:,;!"]+'
PRONOUN_PICK_RE = ur'\W[Bb]en|\W[Ss]en|\W[Bb]iz|\W[Ss]iz'


def complexity_features(text, word_filter):
    sentences = re.findall(SENTENCE_RE, text)
    numSentences = len(sentences)
    numWords = len([word for word in re.split(WORD_SPLITBY_RE, text) if word_filter(word)])
    return {'numWords': numWords, 'numSentences': numSentences}


def pronoun_tokens(text, word_normalizer):
    return {'pronouns': Counter(map(word_normalizer, re.findall(PRONOUN_PICK_RE, text)))}


def character_ngrams(text, n):
    return {'character_ngrams': Counter([''.join([t for t in islice(text, i, n+i)]) for i in range(len(text) - n + 1)])}


def tokenize(text, word_normalizer, word_filter, ngraml):
    tokens = re.split(WORD_SPLITBY_RE, text)
    unigrams = [word_normalizer(token) for token in tokens if word_filter(token)]
    if ngraml is not None:
        ngrams = [tuple([t for t in islice(unigrams, i, n+i)])
              for i in range(len(unigrams) - ngraml[1]) for n in range(*ngraml)]
        return {'unigrams': Counter(unigrams), 'ngrams': Counter(ngrams)}
    else: return {'unigrams': Counter(unigrams)}


def process(path, text_normalizer, word_normalizer, word_filter, ngraml = None,
            pronouns = True, char_ngrams = True, char_ngraml = 5, complexity = True):
    document = {'multinomial': {}, 'count': {}}
    text = text_normalizer(codecs.open(path, encoding='windows-1254').read())
    document['multinomial'].update(tokenize(text, word_normalizer, word_filter, ngraml))
    if pronouns:
        document['multinomial'].update(pronoun_tokens(text, word_normalizer))
    if char_ngrams:
        document['multinomial'].update(character_ngrams(text, char_ngraml))
    if complexity:
        document['count'].update(complexity_features(text, word_filter))

    return document

if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    l = lambda s: s.lower()
    f = lambda s: len(s) > 1 and re.search(r'\d', s) is None
    t = lambda w: w[:w.find(r"'")] if re.search(r"'", w) is not None else w
    doc = process(p, l, t, f, ngraml=(2,3))

    k = 'multinomial'
    print 'Multinomial features:'
    for modality in doc[k]:
        print modality
        for w in doc[k][modality]: print w, doc[k][modality][w]

    k = 'count'
    print 'Count features:'
    for modality in doc[k]:
        print modality, doc[k][modality]


