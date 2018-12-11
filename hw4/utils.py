import numpy as np
from math import ceil

def clean_data(path_to_data):
    content = load_data(path_to_data)

    print(''.join(sorted(set(content)))) # '\t\n !"(),-.0123456789:;?N\xa0«»АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЬЭЯабвгдежзийклмнопрстуфхцчшщъыьэюя–'
    content = content.replace('\t', ' ')
    content = content.replace('\xa0', ' ')
    content = content.replace('«', '\"')
    content = content.replace('»', '\"')
    content = content.replace('–', '-')
    vocabulary = ''.join(sorted(set(content))) # '\n !"(),-.0123456789:;?NАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЬЭЯабвгдежзийклмнопрстуфхцчшщъыьэюя'

    with open(path_to_data, 'w') as dest:
        dest.write(content)

    return vocabulary

def process_data(vocabulary=None, content=None, path_to_data='./data/char-rnn'):
    if content == None:
        content = load_data(path_to_data)
    if vocabulary == None:
        vocabulary = clean_data(path_to_data)

    vocab_to_int = {c: i for i, c in enumerate(vocabulary)}
    int_to_vocab = dict(enumerate(vocabulary))
    encoded_content = np.array([vocab_to_int[c] for c in content], dtype=np.int32)

    return encoded_content, int_to_vocab

def load_data(path_to_data):    
    with open(path_to_data, 'r') as source:
        content = source.read()
    return content

def split_data(data, seq_size):
    count = len(data) // seq_size
    validation_count = ceil(count * 0.1)
    validation_ends = validation_count * seq_size
    validation_data = data[:validation_ends]
    validation_labels = data[1:validation_ends + 1] # index out of range can happen
    train_count = count - validation_count
    train_ends = validation_ends + train_count * seq_size
    train_data = data[validation_ends:train_ends]
    train_labels = data[validation_ends + 1:train_ends + 1] # index out of range can happen

    return np.array(np.split(train_data, train_count)), np.array(np.split(train_labels, train_count)), np.array(np.split(validation_data, validation_count)), np.array(np.split(validation_labels, validation_count))
