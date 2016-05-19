## Platform and Requirements
Tested on python 2.7, Anaconda Interpreter.
Requires numpy

## Data
The scripts are written for and tested on the METU-Sabanci Turkish Treebank <cite>[1]</cite><cite>[2]</cite>

## Usage
### Training an HMM for Part-of-Speech Tagging:
```shell
python train_hmm_tagger.py -h
```
```
usage: train_hmm_tagger.py [-h] TRAIN {postag,cpostag} OUT

positional arguments:
  TRAIN             Training file in conll format
  {postag,cpostag}  Tag type
  OUT               Destination for the trained model output.

optional arguments:
  -h, --help        show this help message and exit
```

### POS-tagging data
```shell
python hmm_tagger.py -h
```
```
usage: hmm_tagger.py [-h] TEST OUT MODEL

positional arguments:
  TEST        Test file
  OUT         Output path for tagged test data to be written
  MODEL       Path for previously trained model

optional arguments:
  -h, --help        show this help message and exit


```

### Evaluating a POS-tagged data with gold standard
```shell
python hw1.py --train /path/to/training/dir --test /path/to/test/dir
```
```
usage: test_hmm_tagger.py [-h] OUT GOLD MODEL {postag,cpostag}

positional arguments:
  OUT               Tagged output to be evaluated
  GOLD              Gold standard data in conll format
  MODEL             Path for previously trained model, this is needed to get word and tag dictionaries
  {postag,cpostag}  Tag type.
                    We need this, in order to extract the correct tag from the gold_standard

optional arguments:
  -h, --help        show this help message and exit

``` 
The script will report the confusion matrix and the classifier accuracy for:
1. all words in the test set
2. only words in the test set those were also observed in the training set

It might be non-obvious why we need serialized model, since we already have tagged output. 
It is because we need the dictionary to identify whether a word is unknown to the model.


### An Example Run
1. Train. -- The third argument is the target for the model to be serialized
```
python train_hmm_tagger.py /path/to/train/file.conll postag /path/to/model/dir
```
2. POS-tag -- with previously trained model:
- The tagged output will be written to a file, and enclosing directory should exist
- The test file to tag can as well be a validation file, the script will ignore the correct assignments in that case
```
python train_hmm_tagger.py /path/to/test/testfile.conll /path/for/tagged/output.txt /path/to/trained/model/dir
```
3. Evaluate -- compare the tagged output with the gold standard file This will write a confusion matrix
```
python test_hmm_tagger.py /path/for/tagged/output.txt /path/for/gold/standard/file.conll /path/to/trained/model/dir postag
```


[1]:Kemal Oflazer, Bilge Say, Dilek Zeynep Hakkani-Tür, Gökhan Tür,
"Building a Turkish Treebank", Invited chapter in "Building and
Exploiting Syntactically-annotated Corpora", Anne Abeille Editor,
Kluwer Academic Publishers, 2003.

[2]:Nart B. Atalay, Kemal Oflazer, Bilge Say, "The Annotation Process in
the Turkish Treebank", in "Proceedings of the EACL Workshop on
Linguistically Interpreted Corpora - LINC", April 13-14, 2003,
Budapest, Hungary
