from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import ray
import model

parser = argparse.ArgumentParser(description="Run the synchronous parameter " "server example.")
parser.add_argument("--num-workers", default=4, type=int, help="The number of workers to use.")
parser.add_argument("--redis-address", default=None, type=str, help="The Redis address of the cluster.")


@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        self.net = model.simple(learning_rate=learning_rate)

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()


@ray.remote
class Worker(object):
    def __init__(self, worker_index, batch_size=50):
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.ds = model.load_data()
        self.net = model.simple()

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.ds.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(redis_address=args.redis_address)

    # Create a parameter server.
    net = model.simple()
    ps = ParameterServer.remote(1e-4 * args.num_workers)

    # Create workers.
    workers = [Worker.remote(worker_index)
               for worker_index in range(args.num_workers)]

    # Download Data.
    ds = model.load_data()

    i = 0
    epoch = 0
    current_weights = ps.get_weights.remote()
    while True:
        # Compute and apply gradients.
        gradients = [worker.compute_gradients.remote(current_weights)
                     for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        if i // ds.train.num_examples == epoch:
            # Evaluate the current model.
            net.variables.set_flat(ray.get(current_weights))
            test_xs, test_ys = ds.test.next_batch(ds.test.num_examples)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            print("Epoch {}: accuracy is {}".format(epoch, accuracy))
            epoch += 1
        i += 50

