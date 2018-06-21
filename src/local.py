import argparse
import sys
import dataset
import tensorflow as tf

FLAGS = None

def main(_):
    ds = dataset.read_data_sets(FLAGS.data_dir)

    x = tf.placeholder(tf.float32, [None, 500])
    w = tf.Variable(tf.zeros([500, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    # y = tf.matmul(x, w) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(1000):
        batch_xs, batch_ys = ds.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: ds.test.features, y_: ds.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/', help='Directory for storing data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

