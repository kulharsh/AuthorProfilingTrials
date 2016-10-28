import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    #positive_examples = [s.strip() for s in positive_examples]
    #negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    #negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    male_examples = list(open("./data/gender-UserModelling/M.txt","r").readlines())
    male_examples = [s.strip() for s in male_examples]
    female_examples = list(open("./data/gender-UserModelling/F.txt","r").readlines())
    female_examples = [s.strip() for s in female_examples]
    '''class1_examples = list(open("./data/age-UserModelling/18-24.txt","r").readlines())
    class1_examples = [s.strip() for s in class1_examples]
    class2_examples = list(open("./data/age-UserModelling/25-34.txt","r").readlines())
    class2_examples = [s.strip() for s in class2_examples]
    class3_examples = list(open("./data/age-UserModelling/35-49.txt","r").readlines())
    class3_examples = [s.strip() for s in class3_examples]
    class4_examples = list(open("./data/age-UserModelling/50-XX.txt","r").readlines())
    class4_examples = [s.strip() for s in class4_examples]'''
    
    #x_text = class1_examples + class2_examples + class3_examples + class4_examples 
    x_text = male_examples + female_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    positive_labels = [[0, 1] for _ in male_examples]
    negative_labels = [[1, 0] for _ in female_examples]
    
    #class1_labels = [[0, 0, 0, 1] for _ in class1_examples]
    #class2_labels = [[0, 0, 1, 0] for _ in class2_examples]
    #class3_labels = [[0, 1, 0, 0] for _ in class3_examples]
    #class4_labels = [[1, 0, 0, 0] for _ in class4_examples]
    
    #y = np.concatenate([class1_labels, class2_labels, class3_labels, class4_labels], 0)
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
