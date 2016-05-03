from corpus import CorpusFactory
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("TRAIN", help="Training file in conll format")
    parser.add_argument("TAG", default="postag", choices=['postag', 'cpostag'], help="Tag type")
    parser.add_argument("OUT", help="Destination for the trained model output.")

    args = parser.parse_args()

    tr = args.TRAIN
    model_path = args.OUT
    tagtype = args.TAG

    corpus = CorpusFactory().train(tr, tagtype=tagtype)
    corpus.persist(model_path)