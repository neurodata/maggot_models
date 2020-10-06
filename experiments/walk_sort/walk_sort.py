#%%
import pickle
import time
from itertools import chain

import colorcet as cc
from graspy.cluster.gclust import GaussianCluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.cluster import AutoGMMCluster
from graspologic.utils import symmetrize
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from scipy.cluster.hierarchy import linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from src.data import DATA_VERSION, DATA_DIR
from src.visualization import CLASS_COLOR_DICT
from pathlib import Path
from src.io import savecsv, savefig

# DATA_VERSION =
meta_loc = f"{DATA_DIR}/{DATA_VERSION}/meta_data.csv"

meta = pd.read_csv(meta_loc, index_col=0)

save_path = Path("maggot_models/experiments/matched_subgraph_omni_cluster/")


def stashfig(name, **kws):
    savefig(name, pathname=save_path / "figs", fmt="pdf", save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, pathname=save_path / "outs", **kws)


# %%

np.random.seed(8888)

test_file = "bert/experiments/pathCA/walks/walks-Gad.txt"

with open(test_file, "r") as f:
    paths = f.read().splitlines()
print(f"# of paths: {len(paths)}")

paths = list(set(paths))
paths.remove("")
print(f"# of paths after removing duplicates: {len(paths)}")

n_subsample = len(paths)  # 2 ** 14
choice_inds = np.random.choice(len(paths), n_subsample, replace=False)
new_paths = []
for i in range(len(paths)):
    if i in choice_inds:
        new_paths.append(paths[i])
paths = new_paths

print(f"# of paths after subsampling: {len(paths)}")
paths = [path.split(" ") for path in paths]
paths = [[int(node) for node in path] for path in paths]
all_nodes = set()
[[all_nodes.add(node) for node in path] for path in paths]
uni_nodes = np.unique(list(all_nodes))
ind_map = dict(zip(uni_nodes, range(len(uni_nodes))))
tokenized_paths = [list(map(ind_map.get, path)) for path in paths]


# %%

node_visits = {}
for path in paths:
    for i, node in enumerate(path):
        if node not in node_visits:
            node_visits[node] = []
        node_visits[node].append(i / (len(path) - 1))

median_node_visits = {}
for node in uni_nodes:
    median_node_visits[node] = np.median(node_visits[node])
meta["median_node_visits"] = meta.index.map(median_node_visits)

median_class_visits = {}
for node_class in meta["merge_class"].unique():
    nodes = meta[meta["merge_class"] == node_class].index
    all_visits_in_class = list(map(node_visits.get, nodes))
    all_visits_in_class = [item for item in all_visits_in_class if item is not None]
    all_visits_flat = list(chain.from_iterable(all_visits_in_class))
    median_class_visits[node_class] = np.median(all_visits_flat)
meta["median_class_visits"] = meta["merge_class"].map(median_class_visits)

meta.to_csv("bert/data/2020-06-10/meta_data_w_order.csv")

# %%

sort_meta = meta.copy()
sort_meta = sort_meta[~sort_meta["median_node_visits"].isna()]
sort_meta.sort_values(
    ["median_class_visits", "merge_class", "median_node_visits"], inplace=True
)


sort_meta["ind"] = range(len(sort_meta))
color_dict = CLASS_COLOR_DICT
classes = sort_meta["merge_class"].values
uni_classes = np.unique(sort_meta["merge_class"])
class_map = dict(zip(uni_classes, range(len(uni_classes))))
color_sorted = np.vectorize(color_dict.get)(uni_classes)
lc = ListedColormap(color_sorted)
class_indicator = np.vectorize(class_map.get)(classes)
class_indicator = class_indicator.reshape(len(classes), 1)

fig, ax = plt.subplots(1, 1, figsize=(1, 10))
sns.heatmap(
    class_indicator,
    cmap=lc,
    cbar=False,
    yticklabels=False,
    # xticklabels=False,
    square=False,
    ax=ax,
)
ax.axis("off")
stashfig("class-rw-order-heatmap")
