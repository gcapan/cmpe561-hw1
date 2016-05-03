from corpus import CorpusFactory
import evaluate as ev
from evaluate import latex_table as report
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("OUT", help="Tagged output to be evaluated")
    parser.add_argument("GOLD", help="Gold standard data in conll format")
    parser.add_argument("MODEL",
                        help="Path for previously trained model, this is needed to get word and tag dictionaries")
    parser.add_argument("TAG", default="postag", choices=['postag', 'cpostag'],
                        help="Tag type.\n"
                             "We need this, in order to extract the correct tag from the gold_standard")
    args = parser.parse_args()
    model = args.MODEL
    output_file = args.OUT
    gold_file = args.GOLD
    tagtype = args.TAG

    fac = CorpusFactory()
    c = fac.load_model(model)
    tag_dictionary_t = dict([(c.tag_dictionary[t], t) for t in c.tag_dictionary])

    predicted_tags, yseqs = fac.load_out(c, output_file)
    _, true_tags, _, _ = fac.load_validation(model, gold_file, tagtype)
    confusion_all, accuracy_all, confusion_known, accuracy_known = \
        ev.confusion(true_tags, predicted_tags, yseqs, tag_dictionary_t, c.default,
                     ignored_classes=[c.start_tag, c.end_tag])

    print "Using", tagtype,"in",gold_file,"as true class"
    print "Accuracy on known words:",accuracy_known
    print "Confusion matrix:",confusion_known
    print
    print "Overall accuracy:",accuracy_all
    print "Confusion matrix:",confusion_all

    report(confusion_all, tag_dictionary_t)

    print

    report(confusion_known, tag_dictionary_t)