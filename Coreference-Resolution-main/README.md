# Coreference-Resolution

## Description
This is an implementation of the multi-pass sieve for coreference resolution described by [Raghunathan et al. (2010)](https://aclanthology.org/D10-1048/). It applies multiple sieves to clusters of referential
expressions in order to link coreferential NP.<br>
This program takes files from the Ontonotes corpus in the conll format as input. Find out more [here](https://cemantix.org/data/ontonotes.html).

## Requirements
Written with Python 3.8.5.<br>
See `requirements.txt`. <br>
Download `nltk`'s stopword corpus by running:<br>
```
>>> import nltk
>>> nltk.download('stopwords')
```


## How to use
Move your directory containing files from the Ontonotes corpus in the conll-format into this directory.
<br>
Run:<br>
```main.py [-h] [--config [CONFIG]] [--ext [EXT]] [--lang [LANG]] in_dir out_dir```<br>

+ `in_dir`: The directory containing the conll files from the Ontonotes corpus. Subdirectories will also be searched.
+ `out_dir`: A name for the directory where output files should be stored.
+ `--config`: The name of the config file where sieves are specified. This defaults to `config.txt`.
+ `--ext`: The extension files should have. This defaults to `conll`. If you only want to extract gold annotated file, set this to `gold_conll`.
+ `--lang`: A subdirectory in `in_dir` from which files should be extracted. Set this to `english` to only extract english files from nested Ontonotes corpus. Per default all subdirectories will be searched.

Examples:<br>
`main.py corpus output`<br>
`main.py --ext gold_conll corpus output`<br>
`main.py --ext gold_conll --lang english nested_corpus output`


### Adjust Sieves
The sieves and their order are specified in `config.txt`. The values specify the order in which the sieves are applied.
To exclude a sieve, set its value to -1.<br>
Example:<br>
```
[Sieves]
Exact_Match_Sieve = 1
Precise_Constructs_Sieve = 2
Strict_Head_Match_Sieve = 3
Strict_Head_Relax_Modifiers = 4
Strict_Head_Relax_Inclusion = -1
```

### Interpret Output
For each document from which coreference information was extracted, there is one file in the output folder. The first line contains the path to the original file.
After that follow clusters of coreferential mentions. Clusters are seperated by `-;-`. <br>
Each line in a cluster represents a mention. The first column is a 3-tuple where the 
first element is the index of the sentence in which the mention appears. The second element is the start index, the third the end index of the mention
in the respective sentence.<br>
For example, assume the following sentence is at index 3:<br>
The<sub>0</sub> dog<sub>1</sub> is<sub>2</sub> happy<sub>3</sub> about<sub>4</sub> his<sub>5</sub> new<sub>6</sub> toy<sub>7</sub><br>
Then the mention "the dog" would have the 3-tuple (3,0,2), the mention "his new toy" would be (3,5,8) and the mention "his" would be (3,5,6).<br>
The second column is the string of the mention.<br>
Example: <br>
```
path/to/example/file.conll
(1,2,3);man
(4,5,6);he
-;-
(7,8,9);dog
(10,11,12);it
```
<br>
Additionally, the output directory contains a file `_summary.csv` that lists the evaluation metrics precision, recall and f1 for each file.

## About

Author: Katja Konermann (katja.konermann@uni-potsdam.de)<br>
Course: Programmierung II<br>
Summer semester 2021