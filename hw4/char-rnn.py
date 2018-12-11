import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import process_data, load_data, split_data, clean_data
from generator import generate_with_model_located_in

tf.logging.set_verbosity(tf.logging.INFO)
DATA_PATH = './data/char-rnn/master_and_margarita.txt'
MODEL_DIR = './model/char-rnn'
EARLY_STOPPING_THRESHOLD = 5
batch_size = 300 # str_cnt
seq_size = 100 # str_len
embedding_size = 128
cell_size = 256
lstm_stack_size = 2
learning_rate = 0.001
clip_norm = 5.0
vocabulary = clean_data(DATA_PATH)
vocab_size = len(vocabulary)


def network(features, mode):
    # Embedding layer
    # Input Tensor Shape: [batch_size, seq_size]
    # Output Tensor Shape: [batch_size , seq_size, vocab_size]
    input_layer = tf.contrib.layers.embed_sequence(features['x'], vocab_size, embedding_size)

    # LSTM layer
    # Consists in lstm_stack_size LSTM cells which consists in cell_size units with dropout
    # Input Tensor Shape: [batch_size , seq_size, vocab_size]
    def lstm_cell(cell_size):
        cell = tf.contrib.rnn.LSTMCell(cell_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.6)
        return cell
    
    cells = tf.contrib.rnn.MultiRNNCell([lstm_cell(cell_size) for _ in range(lstm_stack_size)])
    initial_state = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        initial_state = cells.zero_state(tf.shape(input_layer)[0], dtype=tf.float32)

    # Magic TODO: describe
    # Output Tensor Shape: [batch_size, seq_size, cell_size]
    rnn_out, _ = tf.nn.dynamic_rnn(cells, input_layer, initial_state=initial_state, dtype=tf.float32) # do not save state because didn't find possibility to use it

    # Logits layer
    # Input Tensor Shape: [batch_size, seq_size, cell_size]
    # Output Tensor Shape: [batch_size, seq_size, vocab_size]
    logits = tf.layers.dense(inputs=rnn_out, units=vocab_size)
    
    return logits


def char_rnn_model_fn(features, labels, mode):
    logits = network(features, mode)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': tf.nn.softmax(logits[:,-1,:])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) # shape: [batch_size, seq_size]
    loss = tf.reduce_mean(loss, axis=1) # shape: [batch_size]
    loss = tf.reduce_mean(loss) # shape: 1

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=learning_rate,
            optimizer='Adam',
            clip_gradients=clip_norm)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss)


def main(unused_argv):
    # Load data
    data, _ = process_data(path_to_data=DATA_PATH, vocabulary=vocabulary)
    train_data, train_labels, validation_data, validation_labels = split_data(data, seq_size)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=char_rnn_model_fn,
        model_dir=MODEL_DIR)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

    # Test the model and print results
    validate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': validation_data},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)

    best_model_path = None
    best_loss = 100.0
    degradation_block_cnt = 0
    for _ in range(20):
        classifier.train(
            input_fn=train_input_fn,
            steps=100)

        intermediate_results = classifier.evaluate(
            input_fn=validate_input_fn)
        current_loss = intermediate_results['loss']
        if current_loss >= best_loss:
            degradation_block_cnt += 1
            print('\nDegradation detected: last {} blocks loss increases. Best: {}, current: {}\n'.format(degradation_block_cnt, best_loss, current_loss))
        else:
            best_loss = current_loss
            print('\nLoss decreases: now best is {}\n'.format(best_loss))
            degradation_block_cnt = 0
            best_model_path = classifier.export_savedmodel(
                MODEL_DIR,
                serving_input_receiver_fn=serving_input_receiver_fn)
        if degradation_block_cnt >= EARLY_STOPPING_THRESHOLD:
            print('\nEarly stopped because degradation block count exceeded threshold. Best model has loss {} and is located under {}\n'.format(best_loss, best_model_path))
            break

    final_results = generate_with_model_located_in(best_model_path)
    print('\nBest model located under {}. Generated text: \n {}'.format(best_model_path, final_results))

def serving_input_receiver_fn():
    inputs = {
        'x': tf.placeholder(tf.int32, [None, None])
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


if __name__ == '__main__':
    tf.app.run()
