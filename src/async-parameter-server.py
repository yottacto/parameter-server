from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import ray
import model

# selected_model = model.multilayer_perceptron
selected_model = model.three_layer_perceptron

parser = argparse.ArgumentParser(description="Run the asynchronous parameter server example.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--batch-size", default=64, type=int,
                    help="Batch size.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value

    def pull(self, keys):
        return [self.weights[key] for key in keys]


@ray.remote
def worker_task(ps, worker_index, num_workers, batch_size=64):
    # Download ds.
    ds = model.load_data()

    # Initialize the model.
    net = selected_model()
    keys = net.get_weights()[0]
    while True:
        # Get the current weights from the parameter server.
        weights = ray.get(ps.pull.remote(keys))
        net.set_weights(keys, weights)

        # Compute an update and push it to the parameter server.
        xs, ys = ds.train.next_batch(batch_size)
        gradients = net.compute_update(xs, ys)
        ps.push.remote(keys, gradients)


@ray.remote
def split_batch_worker_task(ps, worker_index, num_workers, batch_size=64):
    # Download ds.
    ds = model.load_data()

    # Initialize the model.
    net = selected_model()
    keys = net.get_weights()[0]
    block_size = batch_size // num_workers
    start = worker_index * block_size
    end = batch_size if worker_index == num_workers - 1 else start + block_size

    while True:
        # Get the current weights from the parameter server.
        weights = ray.get(ps.pull.remote(keys))
        net.set_weights(keys, weights)

        # Compute an update and push it to the parameter server.
        xs, ys = ds.train.next_batch(batch_size)
        xs = xs[start : end]
        ys = ys[start : end]
        gradients = net.compute_update(xs, ys)
        ps.push.remote(keys, gradients)


if __name__ == "__main__":
    preparation_start = time.time()

    args = parser.parse_args()

    ray.init(redis_address=args.redis_address)

    # Create a parameter server with some random weights.
    net = selected_model()
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)

    # Start some training tasks.
    batch_size = args.batch_size
    worker_tasks = [worker_task.remote(ps, i, args.num_workers, batch_size)
            for i in range(args.num_workers)]

    # Download ds.
    ds = model.load_data()

    i = 0
    while i <= 10:
        # Get and evaluate the current model.
        current_weights = ray.get(ps.pull.remote(all_keys))
        net.set_weights(all_keys, current_weights)
        test_xs, test_ys = ds.test.next_batch(1000)
        accuracy = net.compute_accuracy(test_xs, test_ys)

        if i == 0:
            tot_start = time.time()
            iteration_start = tot_start
            print("Preparation time is {}s".format(time.time() - preparation_start))
        else:
            print("Iteration {}: accuracy is {}, time is {}s".format
                    (i, accuracy, time.time() - iteration_start))

        i += 1
        time.sleep(1)
        iteration_start = time.time()

    print("tot time is {} s".format(time.time() - tot_start))

