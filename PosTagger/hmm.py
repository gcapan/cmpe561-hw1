'''
encapsulates transition and emission probabilities of an HMM
contains computation of most likely state sequence given an observation sequence as a method
'''
import numpy as np
class HMM:
    def __init__(self, transitions, word_likelihoods):
        '''
        :param transitions: (note the indexing) transition matrix A, where A_{ij} = P(x_t = i|x_t-1 = j)
        :param word_likelihoods: (note the indexing) observation matrix B, where B_{ij} = P(w_t = i|x_t = j)
         I will assume that the observation space
         contains 2 special symbols, $start$ and $end$, and B_{$start$,0} = B{$end$, N} = 1,
         where 0 and N indexes start and end states, respectively.
         Hence there are two special start and end states
        '''

        # state transitions
        self.A = transitions
        # observations given states
        self.B = word_likelihoods
        self.N = np.shape(self.A)[0]
        pass

    def viterbi(self, y_seq):
        '''
        y_seq is assumed to be augmented, i.e., it starts with $start$ and ends with $ends$
        This assumption on y_seq and A and B (see the doc for the constructor) greatly simplifies the
        Viterbi path implementation.
        :param y_seq: input sequence
        :return: most likely state sequence
        '''
        # max product:
        viterbi = np.zeros((self.N, len(y_seq) + 1))
        backpointer = np.zeros((self.N, len(y_seq) + 1))
        viterbi[0, 0] = 1.
        backpointer[:, 0] = 0
        t = 0
        for w in y_seq:
            t += 1
            viterbi[:, t] = np.max(self.B[w] * self.A * viterbi[:, t-1], axis = 1)
            backpointer[:, t] = np.argmax(self.A * self.B[w] * viterbi[:, t-1], axis = 1)

        x_seq = [np.argmax(backpointer[:, t])]
        for t in range(len(y_seq) - 1, -1, -1):
            x_seq.append(backpointer[x_seq[-1], t])
        x_seq.reverse()
        return x_seq[1:]


class HMMFactory(object):
    def build(self, dictionary, tag_dictionary, sentences):
        N = len(tag_dictionary) + 2
        M = len(dictionary) + 3
        start_tag, end_tag, start_w, end_w, default_w = 0, N-1, 0, M-2, M-1

        transitions = np.zeros((N, N))
        transitions[end_tag, end_tag] = 1.
        observations = np.zeros((M, N))
        observations[start_w, start_tag] = 1.
        observations[end_w, end_tag] = 1.


        for sentence in sentences:
            words_and_tags = [(dictionary[s[0]], tag_dictionary[s[1]]) for s in sentence]
            transitions[words_and_tags[0][1], start_tag] += 1
            transitions[N-1, words_and_tags[-1][1]] += 1

            for w, x in words_and_tags:
                observations[w, x] += 1

            for ((w1, x1), (w2, x2)) in zip(words_and_tags, words_and_tags[1:]):
                transitions[x2, x1] += 1

        # get the most common tag
        most_common_tag = np.argmax(np.sum(transitions, axis=0))
        observations[M-1, most_common_tag] = 1.
        # a column adds up to 1
        transitions = transitions/np.sum(transitions, axis=0)
        observations = observations/np.sum(observations, axis=0)
        return \
            start_w, end_w, default_w, start_tag, end_tag, transitions, observations
