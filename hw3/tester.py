#!/usr/local/env python3

import sys
import numpy as np
from tensorflow.contrib import predictor
from tensorflow.examples.tutorials.mnist import input_data

def test_model_located_in(dir, mnist):
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    predict_fn = predictor.from_saved_model(dir)
    prediction = predict_fn({'x': test_data})
    correct = 0
    for p, a in zip(prediction['classes'], test_labels):
        if p == a:
            correct += 1

    return correct / len(mnist.test.images)


if __name__ == "__main__":
    path_to_model = sys.argv[-1]
    path_to_data = sys.argv[-2]

    data = input_data.read_data_sets(path_to_data, one_hot=False)
    print("Accuracy: ", test_model_located_in(path_to_model, data))
