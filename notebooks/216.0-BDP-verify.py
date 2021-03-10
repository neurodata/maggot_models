#%%
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import pymaid
from src.pymaid import start_instance
from src.visualization import adjplot

#%%

date = "2021-03-09"


def load_meta():
    meta = pd.read_csv(
        f"maggot_models/data/processed/{date}/meta_data.csv", index_col=0
    )
    meta.sort_index(inplace=True)
    return meta


def load_edgelist(graph_type="G"):
    edgelist = pd.read_csv(
        f"maggot_models/data/processed/{date}/{graph_type}_edgelist.txt",
        delimiter=" ",
        header=None,
        names=["source", "target", "weight"],
    )
    return edgelist


def load_networkx(graph_type="G", meta=None):
    edgelist = load_edgelist(graph_type)
    g = nx.from_pandas_edgelist(edgelist, edge_attr="weight", create_using=nx.DiGraph())
    if meta is not None:
        meta_data_dict = meta.to_dict(orient="index")
        nx.set_node_attributes(g, meta_data_dict)
    return g


def load_adjacency(graph_type="G", nodelist=None):
    g = load_networkx(graph_type=graph_type)
    adj = nx.to_numpy_array(g, nodelist=nodelist)
    return adj


meta = load_meta()
ids = meta.index.values


start_instance()
incomplete_ids = pymaid.get_annotated("mw brain incomplete")["skeleton_ids"].values
incomplete_ids = [lst[0] for lst in incomplete_ids]
incomplete_ids
ids = np.setdiff1d(ids, incomplete_ids)

#%%
graph_types = ["Gaa", "Gad", "Gda", "Gdd"]
color_adjs = {}
for graph_type in graph_types:
    color_adj = load_adjacency(graph_type=graph_type, nodelist=ids)
    color_adjs[graph_type] = color_adj


#%%
raw_data_dir = "maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_Brain/"
raw_data_dir = Path(raw_data_dir)
raw_data_dir = raw_data_dir / date

file_names = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]
color_adjs_michael = {}
for graph_type, file_name in zip(graph_types, file_names):
    color_adj = pd.read_csv(raw_data_dir / f"{file_name}.csv", index_col=0, header=0)
    color_adj.columns = color_adj.columns.astype(int)
    color_adj = color_adj.reindex(index=ids, columns=ids, fill_value=0.0)
    print(color_adj.loc[21220994].max())
    print(color_adj[21220994].max())
    color_adjs_michael[graph_type] = color_adj.values

#%%
import seaborn as sns
import matplotlib.pyplot as plt


colors = sns.color_palette("deep")
diffs = {}
sum_adj = np.zeros_like(color_adj)
for graph_type in graph_types:
    my_adj = color_adjs[graph_type]
    michael_adj = color_adjs_michael[graph_type]
    diff = my_adj - michael_adj
    sum_adj += diff

    union = np.count_nonzero(my_adj + michael_adj)
    diff_frac = np.count_nonzero(diff) / union
    print(f"{graph_type} difference fraction: {diff_frac}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    adjplot(diff > 0, plot_type="scattermap", ax=ax, color=colors[0])
    adjplot(diff < 0, plot_type="scattermap", ax=ax, color=colors[1])
    diff_sum = np.sum(np.abs(diff))
    diffs[graph_type] = diff

    color_diff_g = nx.from_numpy_array(diff, create_using=nx.DiGraph())
    node_map = dict(zip(range(len(ids)), ids))
    color_diff_g = nx.relabel_nodes(color_diff_g, node_map)
    out_path = Path(f"maggot_models/data/processed/{date}")
    nx.write_weighted_edgelist(
        color_diff_g, out_path / f"{graph_type}_differences_edgelist.txt"
    )

    nz_inds = list(zip(*np.nonzero(diff)))
    for (pre, post) in nz_inds:
        if pre != 3455 and post != 3455:
            print((pre, post))

    print(diff.min())

print()
print(sum_adj.sum())


#%%
michael_input_output = pd.read_csv(
    raw_data_dir / "input_counts.csv", index_col=0, header=0
)
michael_input_output = michael_input_output.reindex(ids)

#%%
print((meta["axon_input"] == michael_input_output[" axon_inputs"]).all())
#%%
axon_input_diff = meta["axon_input"] - michael_input_output[" axon_inputs"]
# axon_output_diff = meta['de']
axon_input_diff = axon_input_diff[axon_input_diff != 0]
axon_input_diff.index.name = "skid"
axon_input_diff.name = "ben - michael"
axon_input_diff.to_csv(out_path / f"axon_input_differences.txt")


dendrite_input_diff = meta["dendrite_input"] - michael_input_output[" dendrite_inputs"]
# axon_output_diff = meta['de']
dendrite_input_diff = dendrite_input_diff[dendrite_input_diff != 0]
dendrite_input_diff.index.name = "skid"
dendrite_input_diff.name = "ben - michael"
dendrite_input_diff.to_csv(out_path / f"dendrite_input_differences.txt")

# %%
