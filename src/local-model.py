import argparse
import model

parser = argparse.ArgumentParser(description="Run the simple net on local.")

if __name__ == "__main__":
    args = parser.parse_args()

    # Create a parameter server.
    net = model.simple()

    # Download MNIST.
    ds = model.load_data()

    epoch = ds.train.epochs_completed
    while True:
        # Compute and apply gradients.
        xs, ys = ds.train.next_batch(100)
        net.sess.run(net.train_step, feed_dict={net.x: xs, net.y_: ys})

        if ds.train.epochs_completed != epoch:
            # Evaluate the current model.
            test_xs, test_ys = ds.test.next_batch(ds.test.num_examples)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            print("Epoch {}: accuracy is {}".format(epoch, accuracy))
            epoch = ds.train.epochs_completed

