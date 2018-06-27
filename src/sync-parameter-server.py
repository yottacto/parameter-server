from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import ray
import model
import time

selected_model = model.three_layer_perceptron
# selected_model = model.multilayer_perceptron

parser = argparse.ArgumentParser(description="Run the synchronous parameter server example.")
parser.add_argument("--num-workers", default=4, type=int, help="The number of workers to use.")
parser.add_argument("--batch-size", default=256, type=int, help="Batch size.")
parser.add_argument("--redis-address", default=None, type=str, help="The Redis address of the cluster.")


@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        self.net = selected_model(learning_rate=learning_rate)

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()


@ray.remote
class Worker(object):
    def __init__(self, worker_index, num_workers, batch_size=256, learning_rate=1e-4):
        self.worker_index = worker_index
        self.num_workers  = num_workers
        self.batch_size   = batch_size
        self.block_size   = batch_size // num_workers
        self.ds           = model.load_data()
        self.net          = selected_model(learning_rate)

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.ds.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)


@ray.remote
class SplitBatchWorker(object):
    def __init__(self, worker_index, num_workers, batch_size=256, learning_rate=1e-4):
        self.worker_index = worker_index
        self.num_workers  = num_workers
        self.batch_size   = batch_size
        self.block_size   = batch_size // num_workers
        self.start        = self.worker_index * self.block_size
        self.end          = self.batch_size if self.worker_index == self.num_workers - 1 else self.start + self.block_size
        self.ds           = model.load_data()
        self.net          = selected_model(learning_rate)

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.ds.train.next_batch(self.batch_size)
        xs = xs[self.start : self.end]
        ys = ys[self.start : self.end]
        return self.net.compute_gradients(xs, ys)


if __name__ == "__main__":
    preparation_start = time.time()

    args = parser.parse_args()

    learning_rate = 1e-4

    ray.init(redis_address=args.redis_address)

    # Create a parameter server.
    net = selected_model()
    # ps = ParameterServer.remote(1e-4 * args.num_workers)
    ps = ParameterServer.remote(learning_rate)

    # Create workers.
    batch_size = args.batch_size
    workers = [SplitBatchWorker.remote(worker_index, args.num_workers, batch_size, learning_rate)
            for worker_index in range(args.num_workers)]

    # Download Data.
    ds = model.load_data()

    i = 0
    current_weights = ps.get_weights.remote()

    epoch = 0
    while True:
        # Compute and apply gradients.
        gradients = [worker.compute_gradients.remote(current_weights)
                for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        if epoch != i * batch_size // ds.train.num_examples:
            # Evaluate the current model.
            net.variables.set_flat(ray.get(current_weights))
            test_xs, test_ys = ds.test.next_batch(ds.test.num_examples)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            confusion = net.compute_confusion(test_xs, test_ys)

            if epoch == 0:
                tot_start = time.time()
                iteration_start = tot_start
                print("Preparation time is {}s".format(time.time() - preparation_start))
            else:
                print("Epoch {}: accuracy is {}, time is {}s".format
                        (epoch, model.accuracy_from_confusion(confusion),
                            time.time() - iteration_start))

                print("Confusion matrix is:\n{}\n".format(confusion))
                if accuracy > 0.71:
                    break

            iteration_start = time.time()
            epoch = i * batch_size // ds.train.num_examples

        i += 1

    print("tot time is {}s".format(time.time() - tot_start))

