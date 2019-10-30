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
graph_type = "G"
graph = load_networkx(graph_type, version=version)
classes = meta_to_array(graph, "Class")
side_labels = meta_to_array(graph, "Hemisphere")
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
    binarize(adj_df.values),
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    title="Full graph, raw, PTR",
    **heatmap_kws,
)

#%% now plot each color for the mb
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax = ax.ravel()
for i, g in enumerate(graph_types):
    color_graph = load_networkx(g, version=version)
    color_graph_adj = nx.to_pandas_adjacency(color_graph).values
    classes = meta_to_array(graph, "Class")
    simple_classes = np.array(itemgetter(*classes)(name_map))
    side_labels = meta_to_array(graph, "Hemisphere")
    heatmap(
        binarize(color_graph_adj),
        cbar=False,
        hier_label_fontsize=10,
        ax=ax[i],
        inner_hier_labels=simple_classes,
        outer_hier_labels=side_labels,
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

kc_ids = []
graph_data = dict(graph.nodes(data=True))
for node_id, node_dict in graph_data.items():
    if "KC" in node_dict["Class"]:
        kc_ids.append(node_id)
#%%
test_compartment_nodes = test_compartment_nodes + kc_nodes
subgraph_view = nx.induced_subgraph(graph, test_compartment_nodes).copy()
nx.draw_networkx(subgraph_view)
subgraph_adj_df = nx.to_pandas_adjacency(subgraph_view)
#%%
subgraph_classes = meta_to_array(subgraph_view, "Class")
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
simple_subgraph_classes = np.array(itemgetter(*subgraph_classes)(name_map))
hemisphere = meta_to_array(subgraph_view, "Hemisphere")
# this plot is all of the KCs, and the MBONs and MBINs
heatmap(
    subgraph_adj_df.values,
    inner_hier_labels=simple_subgraph_classes,
    outer_hier_labels=hemisphere,
    transform="simple-all",
    sort_nodes=True,
    hier_label_fontsize=10,
    figsize=(30, 30),
)

#%%
def get_targets_in(graph, id, query_nodes):
    """ Find the postsynaptic cells in a particular class given by "query_nodes"
    
    Parameters
    ----------
    graph : [type]
        [description]
    id : [type]
        [description]
    partners : [type]
        [description]
    """
    out_partners = np.array(list(zip(*nx.edges(graph, id)))[1])  # oof
    inds = np.isin(out_partners, query_nodes)
    return out_partners[inds]


# add all of the KCs that the MBINs in this compartment project to.
# these should be A -> D
kcs_from_mbins = []
graph = load_networkx("Gad", version=version)
mbin_ids = ["17068730", "3813487"]
for mbin_id in mbin_ids:
    kc_partners = get_targets_in(graph, mbin_id, kc_ids)
    kcs_from_mbins.append(kc_partners)

# get all of the KCs that connect to the MBINs in this compartment
# these should be A -> A
graph = load_networkx("Gaa", version=version)

for kc_id in kc_ids:
    out_partners = np.array(list(zip(*nx.edges(graph, kc_id)))[1])
    inds = np.isin(mbin_ids, out_partners)  # inds where mbin_ids are in out_partners
    if inds.sum() > 0:
        print(inds.sum())

# remove an#%% Try to load the raw data


# %%
compartment_df
unique_compartments = np.unique(compartment_df["MB Compartment"].values)
unique_compartments

# %% just plot the adjacency matrices for CA
compartment_name = "CA"
mbin_ids = ["17068730", "3813487"]
mbon_ids = ["8877158", "10163418", "17355757", "15617305"]
ca_ids = mbin_ids + mbon_ids + kc_ids

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax = ax.ravel()
graph_type_labels = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]

for i, g in enumerate(graph_types):
    graph = load_networkx(g, version=version)
    subgraph = graph.subgraph(ca_ids).copy()
    adj = nx.to_pandas_adjacency(subgraph).values
    classes = meta_to_array(subgraph, "Class")
    simple_classes = np.array(itemgetter(*classes)(name_map))
    sides = meta_to_array(subgraph, "Hemisphere")
    heatmap(
        binarize(adj),
        inner_hier_labels=simple_classes,
        # inner_hier_labels=sides,
        hier_label_fontsize=12,
        ax=ax[i],
        cbar=False,
        outer_pad=0.7,
        title=graph_type_labels[i],
        title_pad=70,
        sort_nodes=True,
    )
plt.suptitle(compartment_name, fontsize=50)
# %%
