# %% [markdown]
# # Imports
import json
import os
import pickle
import random
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True

sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


def r():
    return random.randint(0, 255)


def extract_ids(lod):
    out_list = []
    for d in lod:
        skel_id = d["skeleton_id"]
        out_list.append(skel_id)
    return out_list


def get_edges(adj):
    adj = adj.copy()
    all_edges = []
    for i in range(adj.shape[0]):
        row = adj[i, :]
        col = adj[:, i]
        col = np.delete(col, i)
        edges = np.concatenate((row, col))
        all_edges.append(edges)
    return all_edges


mg = load_metagraph("Gn", version="2019-12-18")
base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")
pair_df = pd.read_csv(base_path / "pairs/bp-pairs-2020-01-13.csv")
all_cells_file = base_path / "neuron-groups/all-neurons-2020-01-13.json"

skeleton_labels = mg.meta.index.values

# extract valid node pairings
left_nodes = pair_df["leftid"].values
right_nodes = pair_df["rightid"].values

left_right_pairs = list(zip(left_nodes, right_nodes))

with open(all_cells_file) as json_file:
    temp_dict = json.load(json_file)
    all_ids = extract_ids(temp_dict)

mg = load_metagraph("Gn", version="2019-")
all_ids = mg.meta.index.values

for (left, right) in left_right_pairs:
    if left not in all_ids or right not in all_ids:
        print("Missing Pair:")
        print(left)
        print(right)
        if left not in all_ids:
            print(f"Missing ID is: {left}")
        else:
            print(f"Missing ID is: {right}")
        print()

colors = []
ids = []

for (left, right) in left_right_pairs:
    hex_color = "#%02X%02X%02X" % (r(), r(), r())
    colors.append(hex_color)
    colors.append(hex_color)
    ids.append(left)
    ids.append(right)

stashskel("all-paired", ids, colors, colors=colors, palette=None)

left_nodes = []
right_nodes = []
for left, right in left_right_pairs:
    if left in skeleton_labels and right in skeleton_labels:
        left_nodes.append(left)
        right_nodes.append(right)

pair_nodelist = np.concatenate((left_nodes, right_nodes))
not_paired = np.setdiff1d(skeleton_labels, pair_nodelist)
sorted_nodelist = np.concatenate((pair_nodelist, not_paired))

# sort the graph and metadata according to this
sort_map = dict(zip(sorted_nodelist, range(len(sorted_nodelist))))
inv_perm_inds = np.array(itemgetter(*skeleton_labels)(sort_map))
perm_inds = invert_permutation(inv_perm_inds)
mg.reindex(perm_inds)

side_labels = mg["Hemisphere"]
side_labels = side_labels.astype("<U2")
for i, l in enumerate(side_labels):
    if mg.meta.index.values[i] in not_paired:
        side_labels[i] = "U" + l
mg["Hemisphere"] = side_labels
n_pairs = len(left_nodes)
assert (side_labels[:n_pairs] == "L").all()
assert (side_labels[n_pairs : 2 * n_pairs] == "R").all()

# %% [markdown]
# #
mg.verify()

# %% [markdown]
# # do the correlation thingy
adj = mg.adj.copy()

threshold = 0

left_left_adj = adj[:n_pairs, :n_pairs]
left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

# Extract edges
ll_edges = get_edges(left_left_adj)
lr_edges = get_edges(left_right_adj)
rr_edges = get_edges(right_right_adj)
rl_edges = get_edges(right_left_adj)

pair_corrs = []
for i in range(n_pairs):
    left_edge_vec = np.concatenate((ll_edges[i], lr_edges[i]))
    right_edge_vec = np.concatenate((rr_edges[i], rl_edges[i]))
    both_edges = np.stack((left_edge_vec, right_edge_vec), axis=-1)
    avg_edges = np.mean(both_edges, axis=-1)
    # inds = np.where(avg_edges > threshold)[0]

    # check that left and right edge both have
    inds = np.where(
        np.logical_and(left_edge_vec > threshold, right_edge_vec > threshold)
    )[0]
    if len(inds) > 0:
        left_edge_vec = left_edge_vec[inds]
        # left_edge_vec[left_edge_vec > 0] = 1
        right_edge_vec = right_edge_vec[inds]
        # print(left_edge_vec)
        # print(right_edge_vec)
        # right_edge_vec[right_edge_vec > 0] = 1
        R = np.corrcoef(left_edge_vec, right_edge_vec)
        corr = R[0, 1]
        # print(corr)
        # print()
        if i < 40:
            plt.figure()
            plt.scatter(left_edge_vec, right_edge_vec)
            plt.title(corr)
            plt.axis("square")
        # corr = np.count_nonzero(left_edge_vec - right_edge_vec) / len(left_edge_vec)
    else:
        corr = 0
    if np.isnan(corr):
        corr = 0
    pair_corrs.append(corr)
pair_corrs = np.array(pair_corrs)


ground_truth_file = (
    base_path / "neuron-groups/GroundTruth_NeuronPairs_Brain-2019-07-29.csv"
)

ground_truth_df = pd.read_csv(ground_truth_file)
ground_truth_df.set_index("ID", inplace=True)
ground_truth_df.head()
ground_truth_ids = ground_truth_df.index.values
ground_truth_sides = ground_truth_df["Hemisphere"].values

known_inds = []
for cell_id, side in zip(ground_truth_ids, ground_truth_sides):
    if side == " left":
        ind = np.where(mg.meta.index.values == cell_id)[0]
        if len(ind) > 0:
            known_inds.append(ind[0])

not_known_inds = np.setdiff1d(range(n_pairs), known_inds)
new_pair_corrs = pair_corrs[not_known_inds]
truth_pair_corrs = pair_corrs[known_inds]

sns.set_context("talk")
plt.figure(figsize=(10, 5))
sns.distplot(new_pair_corrs, label="New pairs")
sns.distplot(truth_pair_corrs, label="Ground truth")
plt.legend()
plt.title(threshold)
stashfig(f"both-t{threshold}-corr-nodewise")

out_pair_df = pd.DataFrame()
out_pair_df

# %% Look at correlation vs degree
deg_df = mg.calculate_degrees()
plot_df = pd.DataFrame()
total_degree = deg_df["Total degree"].values
plot_df["Mean total degree"] = (
    total_degree[:n_pairs] + total_degree[n_pairs : 2 * n_pairs]
) / 2
plot_df["Correlation"] = pair_corrs
plt.figure(figsize=(10, 5))
sns.scatterplot(data=plot_df, x="Mean total degree", y="Correlation")
stashfig("corr-vs-degree")

sns.jointplot(
    data=plot_df, x="Mean total degree", y="Correlation", kind="hex", height=10
)
stashfig("corr-vs-degree-hex")

# %% [markdown]
# # Find the cells where correlation is < 0
skeleton_labels = mg.meta.index.values[: 2 * n_pairs]
side_labels = mg["Hemisphere"][: 2 * n_pairs]
left_right_pairs = zip(
    skeleton_labels[:n_pairs], skeleton_labels[n_pairs : 2 * n_pairs]
)

colors = []
ids = []

for i, (left, right) in enumerate(left_right_pairs):
    if pair_corrs[i] < 0:
        hex_color = "#%02X%02X%02X" % (r(), r(), r())
        colors.append(hex_color)
        colors.append(hex_color)
        ids.append(left)
        ids.append(right)

stashskel("pairs-low-corr", ids, colors, colors=colors, palette=None)

# %% [markdown]
# # Look at number of disagreements vs degree
prop_disagreements = []
for i in range(n_pairs):
    left_edge_vec = np.concatenate((ll_edges[i], lr_edges[i]))
    right_edge_vec = np.concatenate((rr_edges[i], rl_edges[i]))
    left_edge_vec[left_edge_vec > 0] = 1
    right_edge_vec[right_edge_vec > 0] = 1
    n_disagreement = np.count_nonzero(left_edge_vec - right_edge_vec)
    prop_disagreement = n_disagreement / len(left_edge_vec)
    prop_disagreements.append(prop_disagreement)
prop_disagreements = np.array(prop_disagreements)
plot_df["Prop. disagreements"] = prop_disagreements

sns.jointplot(
    data=plot_df, x="Mean total degree", y="Prop. disagreements", kind="hex", height=10
)
#%%
pair_corrs = []
for i in range(n_pairs):
    left_edge_vec = np.concatenate((ll_edges[i], lr_edges[i]))
    right_edge_vec = np.concatenate((rr_edges[i], rl_edges[i]))
    both_edges = np.stack((left_edge_vec, right_edge_vec), axis=-1)
    avg_edges = np.mean(both_edges, axis=-1)

#%%
left_edges = np.concatenate((left_left_adj.ravel(), left_right_adj.ravel()))
right_edges = np.concatenate((right_right_adj.ravel(), right_left_adj.ravel()))
all_edges = np.stack((left_edges, right_edges), axis=1)
all_edges_sum = np.sum(all_edges, axis=1)
edge_mask = all_edges_sum > 0
all_edges = all_edges[edge_mask]
left_edges = left_edges[edge_mask]
right_edges = right_edges[edge_mask]
mean_edges = np.mean(all_edges, axis=-1)
diff_edges = np.abs(left_edges - right_edges)
plot_df = pd.DataFrame()
plot_df["Mean (L/R) edge"] = mean_edges
plot_df["Diff (L/R) edge"] = diff_edges
plt.figure(figsize=(10, 10))
sns.scatterplot(mean_edges, diff_edges)
sns.jointplot(data=plot_df, x="Mean (L/R) edge", y="Diff (L/R) edge", kind="hex")
plt.figure(figsize=(10, 5))
bins = np.linspace(-1, 40, 41)
sns.distplot(diff_edges, kde=False, norm_hist=False, bins=bins)
sns.jointplot(
    data=plot_df,
    x="Mean (L/R) edge",
    y="Diff (L/R) edge",
    kind="hex",
    xlim=(0, 5),
    ylim=(0, 5),
)

