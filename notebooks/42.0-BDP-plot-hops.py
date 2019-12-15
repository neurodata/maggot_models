# %% [markdown]
# #
import json
import os
import warnings
from operator import itemgetter
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.graph_shortest_path import graph_shortest_path
import networkx as nx

from graspy.cluster import GaussianCluster, AutoGMMCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.data import load_everything
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey
from src.hierarchy import signal_flow

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

BRAIN_VERSION = "2019-12-09"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

BRAIN_VERSION = "2019-12-09"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

SAVEFIGS = True
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

SAVESKELS = False
SAVEOBJS = True

MIN_CLUSTERS = 2
MAX_CLUSTERS = 40
N_INIT = 50
PTR = True
ONLY_RIGHT = True


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


adj, class_labels, side_labels, skeleton_labels = load_everything(
    "Gad",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere"],
    return_ids=True,
)


# select the right hemisphere
if ONLY_RIGHT:
    side = "right hemisphere"
    right_inds = np.where(side_labels == "R")[0]
    adj = adj[np.ix_(right_inds, right_inds)]
    class_labels = class_labels[right_inds]
    skeleton_labels = skeleton_labels[right_inds]
else:
    side = "full brain"

# sort by number of synapses
degrees = adj.sum(axis=0) + adj.sum(axis=1)
sort_inds = np.argsort(degrees)[::-1]
adj = adj[np.ix_(sort_inds, sort_inds)]
class_labels = class_labels[sort_inds]
skeleton_labels = skeleton_labels[sort_inds]

# remove disconnected nodes
adj, lcc_inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[lcc_inds]
skeleton_labels = skeleton_labels[lcc_inds]
bin_adj = binarize(adj)
n_verts=adj.shape[0]
# %% [markdown]
# #
spl_graph = graph_shortest_path(bin_adj, directed=True)
spl_graph[spl_graph == 0] = np.inf
spl_graph[np.diag_indices_from(spl_graph)] = 0


input_inds = []
output_inds = []
for i, label in enumerate(class_labels):
    if "PN" in label:
        input_inds.append(i)
    if "dVNC" in label or "dSEZ" in label:
        output_inds.append(i)

input_inds = np.array(input_inds)
output_inds = np.array(output_inds)

# from i to j

# minimum distance from a PN to each cell
min_hops_from_pn = np.min(spl_graph[input_inds, :], axis=0)

# minimum distance from each cell to a descending
min_hops_to_descend = np.min(spl_graph[:, output_inds], axis=1)

scatter_df = pd.DataFrame()
scatter_df["Min hops from PN"] = min_hops_from_pn
scatter_df["Min hops to descending"] = min_hops_to_descend
# theta = np.arctan2(min_hops_from_pn, min_hops_to_descend)
# mag = np.hypot(min_hops_from_pn, min_hops_to_descend)
# scatter_df["Theta"] = theta
# scatter_df["Mag"] = mag
# scatter_df["Class"] = class_labels
# ax = sns.scatterplot(data=scatter_df, x="Theta", y="Mag")

# plt.figure(figsize=(15, 6))
# ax = sns.stripplot(data=scatter_df, x="Theta", y="Mag", hue="Class", palette='tab20', dodge=True)
# ax.xaxis.set_major_locator(plt.MaxNLocator(3))

plt.figure(figsize=(10, 10))
sns.set_context("talk")
std=0.7
scatter_df["Min hops from PN"] += np.random.uniform(0, std, size=n_verts)
scatter_df["Min hops to descending"] += np.random.uniform(0, std, size=n_verts)
ax = sns.scatterplot(data=scatter_df, x="Min hops from PN", y="Min hops to descending", s=15, linewidth=0, alpha=0.5)
# ax.yaxis.set_majot_formatter(plt.)
ax.axis("equal")
ax.set_xlim((-1, 6))
ax.set_ylim((-1, 6))
ax.set_xticks(range(6))
ax.set_yticks(range(6))
stashfig("min-hop")
# %%
fg = sns.FacetGrid(
    scatter_df,
    col="Class",
    hue="Class",
    col_wrap=5,
    height=4,
    xlim=(-1, 12),
    ylim=(-1, 12),
)
fg.map(sns.heatmap, "Min hops from PN", "Min hops to descending")
stashfig("class-conditional-min-hop")

# %% [markdown]
# #

fig, ax = plt.subplots(5, 4)
ax = ax.ravel()
uni_classes = np.unique(class_labels)
for i, c_label in enumerate(class_labels):
    temp_df = scatter_df[scatter_df["Class"] == c_label]
    mat = temp_df[]
    sns.heatmap(temp)

# %% [markdown]
# #
plt.figure(figsize=(10, 6))
sns.heatmap(spl_graph[input_inds, :])

np.mean(spl_graph[input_inds, :], axis=1)

input_inds[-3]

adj[input_inds[-3], :].max()
adj[:, input_inds[-3]].max()

class_labels[input_inds[-3]]
skeleton_labels[input_inds[-3]]
