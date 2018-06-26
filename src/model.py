# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import tensorflow as tf
import dataset

def load_data(data_dir="/usr/local/data/"):
    return dataset.read_data_sets(data_dir)


class simple(object):
    def __init__(self, learning_rate=1e-4):
        with tf.Graph().as_default():

            # Create the model
            self.x = tf.placeholder(tf.float32, [None, 500])

            # Define loss and optimizer
            self.y = tf.placeholder(tf.float32, [None, 3])

            self.w = tf.Variable(tf.truncated_normal([500, 3], stddev=1/(500+3)))
            self.b = tf.Variable(tf.zeros([3]))
            self.y_ = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)

            with tf.name_scope('loss'):
                self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_), reduction_indices=[1]))

            with tf.name_scope('adam_optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
            self.sess.run(tf.global_variables_initializer())

            # Helper values.

            self.variables = ray.experimental.TensorFlowVariables(
                self.cross_entropy, self.sess)

            self.grads = self.optimizer.compute_gradients(
                self.cross_entropy)
            self.grads_placeholder = [
                (tf.placeholder(tf.float32, shape=grad[1].get_shape()), grad[1])
                for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder)

    def compute_update(self, x, y):
        # TODO(rkn): Computing the weights before and after the training step
        # and taking the diff is awful.
        weights = self.get_weights()[1]
        self.sess.run(self.train_step, feed_dict={self.x: x, self.y: y})
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x, self.y: y})

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self, x, y):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x, self.y: y})

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())


class three_layer_perceptron(object):
    def __init__(self, learning_rate=1e-4):
        with tf.Graph().as_default():

            # Network Parameters
            self.n_hidden_1 = 256 # 1st layer number of features
            self.n_input    = 500 # data input
            self.n_classes  = 3   # total classes

            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])

            # Store layers weight & bias
            self.weights = {
                'h1':  tf.Variable(tf.truncated_normal(
                    [self.n_input,    self.n_hidden_1], stddev=1/(self.n_input+self.n_hidden_1)
                )),
                'out': tf.Variable(tf.truncated_normal(
                    [self.n_hidden_1, self.n_classes], stddev=1/(self.n_hidden_1+self.n_classes)
                ))
            }
            self.biases = {
                'b1':  tf.Variable(tf.zeros([self.n_hidden_1])),
                'out': tf.Variable(tf.zeros([self.n_classes]))
            }


            # Create model
            # Hidden layer with RELU activation
            self.layer_1 = tf.nn.relu(
                tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
            )

            # Output layer with linear activation
            self.out_layer = tf.nn.softmax(
                tf.matmul(self.layer_1, self.weights['out']) + self.biases['out']
            )

            with tf.name_scope('loss'):
                self.cross_entropy = tf.reduce_mean(
                    -tf.reduce_sum(self.y * tf.log(self.out_layer), reduction_indices=[1])
                )

            with tf.name_scope('adam_optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(
                    tf.argmax(self.out_layer, 1), tf.argmax(self.y, 1)
                )
                correct_prediction = tf.cast(correct_prediction, tf.float32)

            self.accuracy = tf.reduce_mean(correct_prediction)

            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
            self.sess.run(tf.global_variables_initializer())

            # Helper values.

            self.variables = ray.experimental.TensorFlowVariables(
                self.cross_entropy, self.sess)

            self.grads = self.optimizer.compute_gradients(
                self.cross_entropy)
            self.grads_placeholder = [
                (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
                for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder)

    def compute_update(self, x, y):
        # TODO(rkn): Computing the weights before and after the training step
        # and taking the diff is awful.
        weights = self.get_weights()[1]
        self.sess.run(self.train_step, feed_dict={self.x: x, self.y: y})
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x, self.y: y})

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self, x, y):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x, self.y: y})

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())


class multilayer_perceptron(object):
    def __init__(self, learning_rate=1e-4):
        with tf.Graph().as_default():

            # Network Parameters
            self.n_hidden_1 = 256 # 1st layer number of features
            self.n_hidden_2 = 256 # 2nd layer number of features
            self.n_input    = 500 # data input
            self.n_classes  = 3   # total classes

            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])

            # Store layers weight & bias
            self.weights = {
                'h1':  tf.Variable(tf.truncated_normal(
                    [self.n_input,    self.n_hidden_1], stddev=1/(self.n_input+self.n_hidden_1)
                )),
                'h2':  tf.Variable(tf.truncated_normal(
                    [self.n_hidden_1, self.n_hidden_2], stddev=1/(self.n_hidden_1+self.n_hidden_2)
                )),
                'out': tf.Variable(tf.truncated_normal(
                    [self.n_hidden_2, self.n_classes], stddev=1/(self.n_hidden_2+self.n_classes)
                ))
            }
            self.biases = {
                'b1':  tf.Variable(tf.zeros([self.n_hidden_1])),
                'b2':  tf.Variable(tf.zeros([self.n_hidden_2])),
                'out': tf.Variable(tf.zeros([self.n_classes]))
            }


            # Create model
            # Hidden layer with RELU activation
            self.layer_1 = tf.nn.relu(
                tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
            )

            self.layer_2 = tf.nn.relu(
                tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2'])
            )

            # Output layer with linear activation
            self.out_layer = tf.nn.softmax(
                tf.matmul(self.layer_2, self.weights['out']) + self.biases['out']
            )

            with tf.name_scope('loss'):
                self.cross_entropy = tf.reduce_mean(
                    -tf.reduce_sum(self.y * tf.log(self.out_layer), reduction_indices=[1])
                )

            with tf.name_scope('adam_optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(
                    tf.argmax(self.out_layer, 1), tf.argmax(self.y, 1)
                )
                correct_prediction = tf.cast(correct_prediction, tf.float32)

            self.accuracy = tf.reduce_mean(correct_prediction)

            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
            self.sess.run(tf.global_variables_initializer())

            # Helper values.

            self.variables = ray.experimental.TensorFlowVariables(
                self.cross_entropy, self.sess)

            self.grads = self.optimizer.compute_gradients(
                self.cross_entropy)
            self.grads_placeholder = [
                (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
                for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder)

    def compute_update(self, x, y):
        # TODO(rkn): Computing the weights before and after the training step
        # and taking the diff is awful.
        weights = self.get_weights()[1]
        self.sess.run(self.train_step, feed_dict={self.x: x, self.y: y})
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x, self.y: y})

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self, x, y):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x, self.y: y})

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())


class simple_cnn(object):
    def __init__(self, learning_rate=1e-4):
        with tf.Graph().as_default():

            # Create the model
            self.x = tf.placeholder(tf.float32, [None, 500])

            # Define loss and optimizer
            self.y_ = tf.placeholder(tf.float32, [None, 3])

            # Build the graph for the deep net
            self.y_conv, self.keep_prob = deepnn(self.x)

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y_, logits=self.y_conv)
            self.cross_entropy = tf.reduce_mean(cross_entropy)

            with tf.name_scope('adam_optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(
                    self.cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.y_conv, 1),
                                              tf.argmax(self.y_, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
            self.sess.run(tf.global_variables_initializer())

            # Helper values.

            self.variables = ray.experimental.TensorFlowVariables(
                self.cross_entropy, self.sess)

            self.grads = self.optimizer.compute_gradients(
                self.cross_entropy)
            self.grads_placeholder = [
                (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
                for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder)

    def compute_update(self, x, y):
        # TODO(rkn): Computing the weights before and after the training step
        # and taking the diff is awful.
        weights = self.get_weights()[1]
        self.sess.run(self.train_step, feed_dict={self.x: x,
                                                  self.y_: y,
                                                  self.keep_prob: 0.5})
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 0.5})

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self, x, y):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is
            the number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with
            values equal to the logits of classifying the digit into one of 10
            classes (the digits 0-9). keep_prob is a scalar placeholder for the
            probability of dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 1, 500, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([1, 25, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_1x4(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([1, 25, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_1x5(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([1 * 25 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 25 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 3])
        b_fc2 = bias_variable([3])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1], padding='SAME')

def max_pool_1x5(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                          strides=[1, 1, 5, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

