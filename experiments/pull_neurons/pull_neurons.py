import datetime
import time
from pathlib import Path
from pickle import dump, load

import numpy as np
import pymaid
from requests.exceptions import ChunkedEncodingError
from src.data import load_maggot_graph
from src.pymaid import start_instance
import matplotlib.pyplot as plt

out_path = Path("maggot_models/experiments/pull_neurons/outs")


t0 = time.time()
start_instance()

mg = load_maggot_graph()
nodes = mg.nodes
ids = [int(i) for i in nodes.index[:100]]

batch_size = 100
max_tries = 5
n_batches = int(np.floor(len(ids) / batch_size))
if len(ids) % n_batches > 0:
    n_batches += 1
print(f"Batch size: {batch_size}")
print(f"Number of batches: {n_batches}")
print(f"Number of neurons: {len(ids)}")
print(f"Batch product: {n_batches * batch_size}\n")

i = 0
currtime = time.time()
nl = pymaid.get_neuron(
    ids[i * batch_size : (i + 1) * batch_size], with_connectors=False
)
print(f"{time.time() - currtime:.3f} seconds elapsed for batch {i}.")
for i in range(1, n_batches):
    currtime = time.time()
    n_tries = 0
    success = False
    while not success and n_tries < max_tries:
        try:
            nl += pymaid.get_neuron(
                ids[i * batch_size : (i + 1) * batch_size], with_connectors=False
            )
            success = True
        except ChunkedEncodingError:
            print(f"Failed pull on batch {i}, trying again...")
            n_tries += 1
    print(f"{time.time() - currtime:.3f} seconds elapsed for batch {i}.")

print("\nPulled all neurons.\b")


print("Pickling...")
currtime = time.time()

with open(out_path / "neurons.pickle", "wb") as f:
    dump(nl, f)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")

with open(out_path / "neurons.pickle", "rb") as f:
    nl = load(f)

nl.plot2d()
plt.show()