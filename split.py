"""
From a parent directory of category directories of input documents, creates 2 directories of training/test data sets
"""
import shutil
import argparse
import os
import random
import json

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("SRC", help="Input directory of all files")
parser.add_argument("DEST", help="Destination parent, in which both training and test collections will be created.")
parser.add_argument("-r", "--ratio", type=int, default=60,
                    help="Size of the training set per class, the result is RATIO percent")
parser.add_argument("-f", "--format", type=str, default='docs', choices=['docs', 'cats'],
                    help="Output format:\n"
                         "docs: a separate file per document, arranged in per-category directories\n"
                         "cats: one json file per category, each file being a json array of "
                         "{doc:[F_NAME], content:[F_CONTENT]} pairs")

args = parser.parse_args()
train = [(c, set(random.sample(os.listdir(os.path.join(args.SRC, c)),
                               int(len(os.listdir(os.path.join(args.SRC, c))) * (0.01 * args.ratio)))))
         for c in os.listdir(args.SRC)]
test = [(c, set(os.listdir(os.path.join(args.SRC, c))) - tr) for (c, tr) in train]

os.mkdir(args.DEST)
os.mkdir(os.path.join(args.DEST, "training"))
os.mkdir(os.path.join(args.DEST, "test"))


def cpf(category, f, parent):
    shutil.copy(os.path.join(args.SRC, category, f), os.path.join(args.DEST, parent, category))


for (tr, ts) in zip(train, test):
    if args.format == 'docs':
        cat = tr[0]
        os.mkdir(os.path.join(args.DEST, "training", cat))
        os.mkdir(os.path.join(args.DEST, "test", cat))
        for f in tr[1]:
            cpf(cat, f, 'training')
        for f in ts[1]:
            cpf(cat, f, 'test')

    elif args.format == 'cats':
        cat = tr[0]
        train_docs, test_docs = [], []
        for f in tr[1]:
            with open(os.path.join(args.SRC, cat, f)) as content:
                train_docs.append({"doc": f, "content": content.read()})
        with open(os.path.join(args.DEST, 'training', cat + ".txt"), "w") as f_out:
            json.dump(train_docs, f_out, indent=0, ensure_ascii=False)

        for f in ts[1]:
            with open(os.path.join(args.SRC, cat, f)) as content:
                test_docs.append({"doc": f, "content": content.read()})
        with open(os.path.join(args.DEST, 'test', cat + ".txt"), "w") as f_out:
            json.dump(test_docs, f_out, indent=0, ensure_ascii=False)
