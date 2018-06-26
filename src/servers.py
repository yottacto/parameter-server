import time
import ray
import argparse

parser = argparse.ArgumentParser(description="List all nodes.")
parser.add_argument("--num-workers", default=1000, type=int, help="The number of workers to use.")
parser.add_argument("--redis-address", default=None, type=str, help="The Redis address of the cluster.")

@ray.remote
def f():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address)

    # Get a list of the IP addresses of the nodes that have joined the cluster.
    while True:
        all_nodes = set(ray.get([f.remote() for _ in range(args.num_workers)]))
        print("There are {} nodes.".format(len(all_nodes)))
        print("IP addresses are {}.".format(all_nodes))
        time.sleep(2)

