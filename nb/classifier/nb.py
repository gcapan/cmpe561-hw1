from collections import defaultdict
from collections import Counter
from itertools import groupby
from operator import itemgetter
from scipy.stats import poisson

import math
import numpy as np

def build(docs_of_classes):
    numDocs = 0
    ret = {}
    for j in docs_of_classes:
        docs_c = docs_of_classes[j]
        c = {}
        c['multinomial'] = defaultdict(Counter)
        c['count'] = defaultdict(list)
        for d in docs_c:
            for f in d['multinomial']:
                c['multinomial'][f] += d['multinomial'][f]
            for r in d['count']:
                c['count'][r].append(d['count'][r])

        Nc = {}
        k = 'multinomial'
        for modality in c[k]:
            Nc[modality] = sum(c[k][modality].values())

        k = 'count'
        Nr = {}
        for modality in c[k]:
            Nr[modality] = {}
            vec = np.array(c[k][modality])
            Nr[modality]['mean'] = np.mean(vec)

        numDocs += len(docs_c)
        ret[j] = len(docs_c), c, Nc, Nr
    vocabulary = defaultdict(Counter)
    for j in ret:
        c = ret[j][1]
        for modality in c['multinomial']:
            vocabulary[modality] = vocabulary[modality] | c['multinomial'][modality]

    return numDocs, vocabulary, ret

def estimate(doc, model, alpha):
    vocabulary = model[1]
    numDocs = model[0]
    classes = model[2]
    scores = {}
    for c in classes:
        log_prior = math.log(classes[c][0] / float(numDocs))
        word_freqs = classes[c][1]
        Nc = classes[c][2]
        Nr = classes[c][3]
        m_evidence = [[doc['multinomial'][m][w] *
              math.log((word_freqs['multinomial'][m][w] + alpha)/(alpha * len(vocabulary[m]) + float(Nc[m])))
              for w in doc['multinomial'][m]]
             for m in doc['multinomial']]
        m_evidence = [sum(modality) for modality in m_evidence]

        if len(Nr) > 0:
            r_evidence = np.array([poisson.logpmf(doc['count'][m], Nr[m]['mean'])
                        for m in doc['count']])
            scores[c] = log_prior + np.sum(m_evidence) + np.sum(r_evidence)
        else:
            scores[c] = log_prior + np.sum(m_evidence)

    return max(scores, key = scores.get)


def evaluate(actual, predicted):
    classes = set(actual)
    prediction_true = [(p, p == a) for (p, a) in zip(predicted, actual)]
    al = [(x, map(itemgetter(1), y)) for x, y in groupby(sorted(prediction_true), itemgetter(0))]
    tp_fp = [(c, (sum(l), len(l) - sum(l))) for c,l in al]
    for c in classes:
        if c not in predicted:
            tp_fp.append((c, (0, 0)))

    #print tp_fp

    micro_averaged_tp_fp = reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]), dict(tp_fp).values())
    #print micro_averaged_tp_fp

    actual_positive = [(a, p == a) for (a, p) in zip(actual, predicted)]
    al = [(x, map(itemgetter(1), y)) for x, y in groupby(sorted(actual_positive), itemgetter(0))]
    tp_fn = [(c, (sum(l), len(l)-sum(l))) for c, l in al]

    micro_averaged_tp_fn = reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]), dict(tp_fn).values())

    precisions = precision(tp_fp, micro_averaged_tp_fp)
    recalls = recall(tp_fn, micro_averaged_tp_fn)
    f1s = f1(precisions, recalls)
    return {'precision': precisions, 'recall': recalls, 'f1': f1s}


def precision(tp_fp, micro_averaged_tp_fp):
    p = [(c, tp/float(tp+fp)) if tp+fp > 0 else (c, 0) for (c, (tp, fp)) in tp_fp ]
    micro_averaged_prec = micro_averaged_tp_fp[0]/float(micro_averaged_tp_fp[0] + micro_averaged_tp_fp[1])
    al = [prec for (c, prec) in p]
    macro_averaged_prec = sum(al) / float(len(al))
    precisions = dict(p)
    precisions['micro-averaged'] = micro_averaged_prec
    precisions['macro-averaged'] = macro_averaged_prec
    return precisions


def recall(tp_fn, micro_averaged_tp_fn):
    r = [(c, tp/float(tp+fn)) for (c, (tp, fn)) in tp_fn]
    micro_averaged_rec = micro_averaged_tp_fn[0]/float(micro_averaged_tp_fn[0] + micro_averaged_tp_fn[1])
    all = [rec for (c, rec) in r]
    macro_averaged_rec = sum(all) / float(len(all))
    recalls = dict(r)
    recalls['micro-averaged'] = micro_averaged_rec
    recalls['macro-averaged'] = macro_averaged_rec
    return recalls


def f1(precisions, recalls):
    f = {}
    for c in precisions:
        if precisions[c] == 0 and recalls[c] == 0: f[c] = 0
        else: f[c] = (2 * precisions[c] * recalls[c])/(precisions[c] + recalls[c])
    all = [f[c] for c in f if c != 'micro-averaged' and c!= 'macro-averaged']
    f['macro-averaged'] = sum(all) / (float(len(all)))
    return f
