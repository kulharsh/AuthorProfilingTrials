"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='/Tmp/bastienf/aclImdb/'
dataset_path = '/home/harshad/author_profiling/RNN/deeplearning/data/aclImdb/'
import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(path):
    sentences = []
    with open('M', 'r') as f:
	for line in f:    	
		sentences.append(line.strip())
    with open('F', 'r') as f:
	for line in f:
	    	sentences.append(line.strip())
    
    sentences = tokenize(sentences)
    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    with open(path, 'r') as f:
	for line in f:	    	
		sentences.append(line.strip())
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train'))

    x_pos = grab_data('M', dictionary)
    x_neg = grab_data('F', dictionary)
    x = x_pos + x_neg
    y = [1] * len(x_pos) + [0] * len(x_neg)
    train_x = x[:100] 
    test_x = x[100:]
    train_y = y[:100] 
    test_y = y[100:]
 	    
    print len(train_x),len(train_y), 'samples in the training set'
    print len(test_x),len(test_y), 'samples in the testing set' 	
    f = open('gender.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('gender.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()
