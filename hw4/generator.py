#!/usr/local/env python3

import sys
import numpy as np
from tensorflow.contrib import predictor

from utils import process_data

def pick_top_n(preds, vocab_size, top_n=3):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def generate_with_model_located_in(dir, init_seq='Разрешите мне присесть?', count=100):
  vocabulary = '\n !"(),-.0123456789:;?NАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЬЭЯабвгдежзийклмнопрстуфхцчшщъыьэюя'
  text_encoded, int_to_vocab = process_data(vocabulary=vocabulary, content=init_seq)
  for _ in range(count):
    generate_fn = predictor.from_saved_model(dir)
    answer = generate_fn({'x': [text_encoded]})
    # symbol_code = np.argmax(answer['probabilities'][0])
    symbol_code = pick_top_n(answer['probabilities'][0], len(vocabulary))
    text_encoded = np.append(text_encoded, symbol_code)
  
  text = '\n===\n'
  for code in text_encoded:
    text += int_to_vocab[code]
  return text


if __name__ == "__main__":
  path_to_model = sys.argv[-3]
  init_seq = sys.argv[-2]
  count = int(sys.argv[-1])
  with open('./output.txt', 'a') as output:
    output.write(generate_with_model_located_in(path_to_model, init_seq, count))
