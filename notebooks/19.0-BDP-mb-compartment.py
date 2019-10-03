#%%
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.plot import heatmap
from graspy.utils import binarize
from src.data import load_networkx
from src.utils import meta_to_array

version = "mb_2019-09-23"


heatmap_kws = dict(
    hier_label_fontsize=16,
    title_pad=200,
    sort_nodes=True,
    transform="simple-all",
    figsize=(20, 20),
    font_scale=1.5,
)
plt.style.use("seaborn-white")
sns.set_palette("deep")
#%% Load and plot the full graph
graph_type = "Gaa"
graph = load_networkx(graph_type, version=version)
classes = meta_to_array(graph, "Class")
hemisphere = meta_to_array(graph, "Hemisphere")
adj_df = nx.to_pandas_adjacency(graph)
name_map = {
    "APL": "APL",
    "Gustatory PN": "PN",
    "KC 1 claw": "KC",
    "KC 2 claw": "KC",
    "KC 3 claw": "KC",
    "KC 4 claw": "KC",
    "KC 5 claw": "KC",
    "KC 6 claw": "KC",
    "KC young": "KC",
    "MBIN": "MBIN",
    "MBON": "MBON",
    "ORN mPN": "PN",
    "ORN uPN": "PN",
    "Unknown PN": "PN",
    "tPN": "PN",
    "vPN": "PN",
}
simple_classes = np.array(itemgetter(*classes)(name_map))

heatmap(
    adj_df.values,
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    title="Full graph, raw, PTR",
    **heatmap_kws,
)

#%% Try to get a compartment out of the full graph
data_path = Path(
    "./maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_MushroomBody"
)

data_date = "2019-09-23"
compartment_file = "MB_compartments.csv"
f = data_path / data_date / compartment_file
compartment_df = pd.read_csv(f)

test_compartment = "CA"
test_compartment_df = compartment_df[
    compartment_df["MB Compartment"] == test_compartment
]
test_compartment_df
#%%
test_compartment_nodes = list(test_compartment_df["ID"].values.astype(str))
test_compartment_nodes

kc_nodes = []
graph_data = dict(graph.nodes(data=True))
for node_id, node_dict in graph_data.items():
    if "KC" in node_dict["Class"]:
        kc_nodes.append(node_id)
#%%
test_compartment_nodes = test_compartment_nodes + kc_nodes
subgraph_view = nx.induced_subgraph(graph, test_compartment_nodes).copy()
nx.draw_networkx(subgraph_view)
subgraph_adj_df = nx.to_pandas_adjacency(subgraph_view)
#%%
heatmap(subgraph_adj_df.values)

#%%
compartment_ids = []

# add all of the KCs that the MBINs talk to 
mbin_ids = ["17068730", "3813487"]
for mbin_id in mbin_ids:
    out_partners = list(zip(*nx.edges(graph, mbin_id)))
    for out_partner in out_partners:
        if out_partner in kc_nodes:
            compartment_ids.append(out_partner)

# remove any 
for 