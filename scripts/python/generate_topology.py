import json
import torch
import os
import sys

from kompil.nn.topology import BASE_TOPOLOGY

TOPOLOGIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "res/topologies/"
)


def to_json(filename: str):
    topology = BASE_TOPOLOGY
    topology_json = json.dumps(topology, indent=4)

    if not filename.endswith(".json"):
        filename += ".json"

    with open(os.path.join(TOPOLOGIES_DIR, filename), "w+") as f:
        f.write(topology_json)


def main(argv):
    to_json(argv[0])


# Generate a json file from a topology
# Usage : py generate_topology.py filname.json
if __name__ == "__main__":
    main(sys.argv[1:])
