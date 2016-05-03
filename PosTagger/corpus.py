# coding=utf-8
import re
import codecs
import json
import os
import numpy as np
from hmm import HMMFactory
START_TOKEN = "$$START_TOKEN$$"
END_TOKEN = "$$END_TOKEN$$"
DEFAULT_TOKEN = "$$UNKNOWN_WORD$$"
START_TAG = "$$START_TAG$$"
END_TAG = "$$END_TAG$$"


class Corpus:
    def __init__(self, start_w, end_w, default_w, start_tag, end_tag,
                 dictionary, tag_dictionary, transitions, observations):
        self.dictionary = dictionary
        self.tag_dictionary = tag_dictionary
        self.transitions = transitions
        self.observations = observations
        self.default = default_w
        self.start_token = start_w
        self.end_token = end_w
        self.start_tag = start_tag
        self.end_tag = end_tag

        dictionary[START_TOKEN] = self.start_token
        dictionary[END_TOKEN] = self.end_token
        dictionary[DEFAULT_TOKEN] = self.default

        tag_dictionary[START_TAG] = self.start_tag
        tag_dictionary[END_TAG] = self.end_tag


    def persist(self, path):
        with codecs.open(os.path.join(path, "dictionary"), "w", encoding="utf-8") as dict_out:
            dict_out.write(json.dumps(self.dictionary))

        with codecs.open(os.path.join(path, "tags"), "w", encoding="utf-8") as tags_out:
            tags_out.write(json.dumps(self.tag_dictionary))

        np.save(os.path.join(path, "observations"), self.observations)
        np.save(os.path.join(path, "transitions"), self.transitions)

    def yseq(self, tokenlist):
        return [self.start_token]+[self.dictionary.get(token, self.default) for token in tokenlist] + [self.end_token]

    def xseq(self, tags):
        return [self.start_tag]+[self.tag_dictionary[t] for t in tags]+[self.end_tag]
    def yseqs(self, sentences):
        return [self.yseq(l) for l in sentences]

    def xseqs(self, sentences):
        return [self.xseq(l) for l in sentences]



class CorpusFactory:
    def train(self, path, tagtype="postag"):
        lines = codecs.open(path, encoding="utf-8").readlines()
        sentences = []
        sep = r'[\s|^]+'
        pin = 3 if tagtype == "postag" else 4
        words = set()
        tags = set()

        first = True
        i = 0
        for line in lines:
            if re.match(sep, line) is None:
                t = line.split("\t")
                word = t[1]
                if word != "_":
                    if first:
                        sentences.append([])
                        first = False
                    tag = t[pin]
                    words.add(word)
                    tags.add(tag)
                    sentences[i].append((t[1], tag))
            else:
                first = True
                i += 1
        dictionary = dict(zip(sorted(list(words)), range(1, len(words)+1)))

        tag_dictionary = dict((zip(tags, range(1, len(tags)+1))))
        # because python v 2.7
        fac = HMMFactory()
        start_w, end_w, default_w, start_tag, end_tag, transitions, observations =\
            fac.build(dictionary, tag_dictionary, sentences)

        return Corpus(start_w, end_w, default_w, start_tag, end_tag,
                      dictionary, tag_dictionary, transitions, observations)

    def load_validation(self, model_path, data_path, tagtype="postag"):
        corpus = self.load_model(model_path)
        lines = codecs.open(data_path, encoding="utf-8").readlines()
        words = []
        tags = []
        sep = r'[\s|^]+'
        pin = 3 if tagtype == "postag" else 4

        first = True
        i = 0
        for line in lines:
            if re.match(sep, line) is None:
                t = line.split("\t")
                word = t[1]
                if word != "_":
                    if first:
                        words.append([])
                        tags.append([])
                        first = False
                    words[i].append(t[1])
                    tags[i].append(t[pin])
            else:
                first = True
                i += 1

        return corpus, corpus.xseqs(tags), corpus.yseqs(words), words

    def load_test(self, model_path, data_path):
        '''
        :param model_path: Previously trained model to be used for tag assignment.
        Used only for the dictionary
        :param data_path: Test data path.
        This does not have to be blind, the function will consider only the word part
        :return: corpus and test sequences, as sequences of word_ids
        '''
        corpus = self.load_model(model_path)
        lines = codecs.open(data_path, encoding="utf-8").readlines()
        words = []
        sep = r'[\s|^]+'

        first = True
        i = 0
        for line in lines:
            if re.match(sep, line) is None:
                t = line.split("\t")
                word = t[1]
                if word != "_":
                    if first:
                        words.append([])
                        first = False
                    words[i].append(t[1])
            else:
                first = True
                i += 1

        return corpus, corpus.yseqs(words), words

    def load_out(self, corpus, outpath):
        predicted_tags = []
        yseqs = []

        lines = codecs.open(outpath, encoding="utf-8").readlines()
        sep = r'[\s|^]+'
        i = 0
        first = True
        for line in lines:
            if re.match(sep, line) is None:
                line = line[:-1]
                t = line.split("|")
                if first:
                    predicted_tags.append([])
                    yseqs.append([])
                    first = False
                predicted_tags[i].append(t[1])
                yseqs[i].append(t[0])
            else:
                first = True
                i += 1
        return corpus.xseqs(predicted_tags), corpus.yseqs(yseqs)


    def load_model(self, path):
        observations = np.load(os.path.join(path, "observations.npy"))
        transitions = np.load(os.path.join(path, "transitions.npy"))
        dictionary = json.load(codecs.open(os.path.join(path, "dictionary"),encoding="utf-8"))
        tag_dictionary = json.load(codecs.open(os.path.join(path, "tags"),encoding="utf-8"))
        return Corpus(dictionary[START_TOKEN], dictionary[END_TOKEN], dictionary[DEFAULT_TOKEN],
                      tag_dictionary[START_TAG], tag_dictionary[END_TAG],
                      dictionary, tag_dictionary, transitions, observations)
