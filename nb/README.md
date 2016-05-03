## Platform and Requirements
Tested on python 2.7, Anaconda Interpreter.
Requires numpy and scipy

## Notes
This is a very inefficient implementation. No vectorization, and a corpus is generated with adding python Counters
Each experiment reloads and renormalizes the input dataset, making it slower.

## Usage
### Splitting a dataset:
```shell
python split.py [SRC] [DEST]
```

There are a bunch of other arguments to play with (such as the training/test split ratio). To see the full documentation,
run it as:
```shell
python split.py --help
```

### Running and Experimenting a NB Classifier:
The following command will take as input the training and test directories, run a bunch of experiments and report the
overall (micro and macro averaged) precision, recall, and f1-measures:
```shell
python hw1.py --train /path/to/training/dir --test /path/to/test/dir
```
The experiments are predefined, and they compare the NB
performance with different feature sets and different smoothing parameters