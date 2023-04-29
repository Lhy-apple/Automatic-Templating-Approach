import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),\+!?\'\`]", " ", string)
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
    string = re.sub(r"\+", " \+ ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def not_empty(s):
    return s and s.strip()

def not_too_long(s):
    if len(s.split(" "))>100:
        return False
    return True

def not_too_short(s):
    if len(s.split(" "))<10:
        return False
    return True


def load_data_and_labels(files,frc):
    """
    Loads from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    labelSize = len(files)#类别数量
    x_text3 = np.array([])
    x_text1 = np.array([])
    x_text2 = np.array([])
    y_label3 = np.array([])
    y_label1 = np.array([])
    y_label2 = np.array([])
    for fileIndex, file in enumerate(files):
        # read all sentences and split each sentence into words
        # sentences = list(open(file, "r").readlines())
        sentences = list(open(file, "rb").readlines())
        sentences = [s.strip() for s in sentences]
        sentences = [clean_str(str(s)) for s in sentences]#数据清洗
        sentences = list(filter(not_empty, sentences))
        sentences = list(filter(not_too_long, sentences))
        sentences = list(filter(not_too_short, sentences))
        # sentences = np.array(sentences)
        temp = np.empty(len(sentences), dtype=object)
        for i, a in enumerate(sentences):
            temp[i] = a
        np.random.shuffle(temp)
        #temp=temp[0:4559]
        print(len(temp))
        sentences = temp[:int(len(temp)*frc)]
        print(len(sentences))
        num1 = int(len(sentences) * 0.6)
        num2 = int(len(sentences) * 0.8)
        x_text1 = np.concatenate([x_text1, sentences[:num1]], 0)  # 多数组拼接
        x_text2 = np.concatenate([x_text2, sentences[num1:num2]], 0)  # 多数组拼接
        x_text3 = np.concatenate([x_text3, sentences[num2:]], 0)  # 多数组拼接
        print(len(x_text1))
        print(len(x_text2))
        print(len(x_text3))
        # generate labels for each sentence
        labels = [ [0 for x in range(labelSize)] for i in range(len(sentences)) ]#[[00000] [00000]]
        for label in labels: label[fileIndex] = 1
        if fileIndex == 0:
            y_label1 = np.array(labels[:num1])
            y_label2 = np.array(labels[num1:num2])
            y_label3 = np.array(labels[num2:])
        else:
            y_label1 = np.concatenate([y_label1, labels[:num1]], 0)
            y_label2 = np.concatenate([y_label2, labels[num1:num2]], 0)
            y_label3 = np.concatenate([y_label3, labels[num2:]], 0)
    return [x_text1, y_label1,x_text2,y_label2,x_text3,y_label3]


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
            shuffle_indices = np.random.permutation(np.arange(data_size))#随机置换
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def seq_length(x_batch):
    real_seq_len = []
    for line in x_batch:
        real_seq_len.append(np.sum(np.sign(line)))
    return real_seq_len