import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tester import test_model_located_in

tf.logging.set_verbosity(tf.logging.INFO)
DATA_DIR = './data/fashion'
MODEL_DIR = './model/fashion/refactoring'
EARLY_STOPPING_THRESHOLD = 5


def network(features, mode):
    layers = []
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    layers.append(tf.reshape(features["x"], [-1, 28, 28, 1]))

    # Convolutional Layer #1
    # Computes 32 features using two 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    layers.append(tf.layers.conv2d(
        inputs=layers[-1],
        filters=32,
        kernel_size=[3, 3],
        padding="same"))

    layers.append(tf.layers.conv2d(
        inputs=layers[-1],
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu))

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    layers.append(tf.layers.max_pooling2d(
        inputs=layers[-1], 
        pool_size=[2, 2],
        strides=2))

    # Convolutional Layer #2
    # Computes 64 features using two 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    layers.append(tf.layers.conv2d(
        inputs=layers[-1],
        filters=64,
        kernel_size=[3, 3],
        padding="same"))

    layers.append(tf.layers.conv2d(
        inputs=layers[-1],
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu))

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    layers.append(tf.layers.max_pooling2d(
        inputs=layers[-1],
        pool_size=[2, 2],
        strides=2))

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    layers.append(tf.reshape(layers[-1], [-1, 7 * 7 * 64]))

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    layers.append(tf.layers.dense(
        inputs=layers[-1],
        units=1024,
        activation=tf.nn.relu))

    # Add dropout operation; 0.6 probability that element will be kept
    layers.append(tf.layers.dropout(
        inputs=layers[-1], 
        rate=0.4, 
        training=mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    layers.append(tf.layers.dense(
        inputs=layers[-1],
        units=10))
    
    return layers


def cnn_model_fn(features, labels, mode):
    layers = network(features, mode)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=layers[-1], axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(layers[-1], name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=layers[-1])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load data
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=False)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    validation_data = mnist.validation.images
    validation_labels = np.asarray(mnist.validation.labels, dtype=np.int32)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=MODEL_DIR)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=400,
        num_epochs=None,
        shuffle=True)

    # Test the model and print results
    validate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': validation_data},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)

    best_model_path = None
    best_accuracy = 0.0
    degradation_block_cnt = 0
    for _ in range(100):
        classifier.train(
            input_fn=train_input_fn,
            steps=100)

        intermediate_results = classifier.evaluate(
            input_fn=validate_input_fn)
        current_accuracy = intermediate_results['accuracy']
        if current_accuracy <= best_accuracy:
            degradation_block_cnt += 1
            print('\nDegradation detected: last {} blocks accuracy decreases. Best: {}, current: {}\n'.format(degradation_block_cnt, best_accuracy, current_accuracy))
        else:
            best_accuracy = current_accuracy
            print('\nAccuracy increases: now best is {}\n'.format(best_accuracy))
            degradation_block_cnt = 0
            best_model_path = classifier.export_savedmodel(
                MODEL_DIR,
                serving_input_receiver_fn=serving_input_receiver_fn)
        if degradation_block_cnt >= EARLY_STOPPING_THRESHOLD:
            print('\nEarly stopped because degradation block count exceeded threshold. Best model has accuracy {} and is located under {}\n'.format(best_accuracy, best_model_path))
            final_results = test_model_located_in(best_model_path, mnist)
            print('\nFinal accuracy: {}\n'.format(final_results))
            # didn't find way to stop estimator, so will exit(0)
            exit(0)

    final_results = test_model_located_in(best_model_path, mnist)
    print('===\nFinal accuracy {}, model is located under {}\n'.format(final_results, best_model_path))

def serving_input_receiver_fn():
    inputs = {
        'x': tf.placeholder(tf.float32, [None, 28 * 28])
        }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


if __name__ == "__main__":
    tf.app.run()
