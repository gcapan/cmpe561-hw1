from corpus import CorpusFactory
from hmm import HMM
import argparse
import codecs

if __name__ == "__main__":
    '''
    A validation dataset can be passed as the test file.
    The script will ignore the the correct test assignments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("TEST", help="Test file")
    parser.add_argument("OUT", help="Output path for tagged test data to be written")
    parser.add_argument("MODEL", help="Path for previously trained model")

    args = parser.parse_args()

    test_data = args.TEST
    model_path = args.MODEL
    output_path = args.OUT

    corpus, yseqs, sentences = CorpusFactory().load_test(model_path, test_data)
    tag_dictionary_t = dict([(corpus.tag_dictionary[t], t) for t in corpus.tag_dictionary])

    hmm = HMM(corpus.transitions, corpus.observations)
    bw = codecs.open(output_path, mode="w", encoding="utf-8")

    for sentence, words in zip(yseqs, sentences):
        tags_predicted = hmm.viterbi(sentence)
        tags_predicted = tags_predicted[1:-1]
        for w, x in zip(words, tags_predicted):
            bw.write(w+"|"+tag_dictionary_t[int(x)]+"\n")
        bw.write("\n")
