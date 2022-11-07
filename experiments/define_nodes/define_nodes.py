#%%

import datetime
import time

import numpy as np
import pandas as pd
from graspologic.utils import is_fully_connected
from src.data import join_node_meta, load_maggot_graph

t0 = time.time()

#%%
def join_index_as_indicator(index, name):
    series = pd.Series(index=index, data=np.ones(len(index), dtype=bool), name=name)
    join_node_meta(series, overwrite=True, fillna=False)


#%%

mg = load_maggot_graph()
nodes = mg.nodes.copy()

nodes = nodes[nodes["brain_and_inputs"] | nodes["accessory_neurons"]]
print("Number of brain, input, and accessory nodes:", len(nodes))
join_index_as_indicator(nodes.index, "considered")

nodes = nodes[~nodes["very_incomplete"]]
nodes = nodes[~nodes["partially_differentiated"]]
nodes = nodes[~nodes["motor"]]
print(
    "Number of nodes after removing incomplete, partially differentiated, and motor neurons:",
    len(nodes),
)
join_index_as_indicator(nodes.index, "selected")

mg = mg.node_subgraph(nodes.index)

mg.to_largest_connected_component()

print("Number of nodes after taking LCC:", len(mg))


print("Is fully connected:", is_fully_connected(mg.sum.adj))

join_index_as_indicator(mg.nodes.index, "selected_lcc")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
