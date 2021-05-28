#%% [markdown]
# # Layouts for an auditory connectome
#%%
import datetime
import time
from pathlib import Path
from adjustText import adjust_text
from graspologic import partition
import matplotlib.pyplot as plt
import colorcet as cc
import h5py
from networkx.generators import directed
import numpy as np
from numpy.lib.utils import source
import pandas as pd
from giskard.plot import graphplot
from graspologic.layouts.colors import _get_colors
from matplotlib.lines import Line2D
from src.io import savefig
from src.visualization.settings import set_theme
from graspologic.utils import symmetrize
from graspologic.partition import leiden, modularity
import networkx as nx
from src.visualization import adjplot, set_theme
from graspologic.utils import pass_to_ranks

t0 = time.time()


def stashfig(name):
    savefig(name, foldername="auditory-layout", pad_inches=0.05)


#%% [markdown]
# ## Load data
#%%
mat_path = Path("maggot_models/data/raw/allAudNeurons_connMat_noDupSyn.mat")
f = h5py.File(mat_path)
adj_key = "cxns_noDup"
label_key = "neuronClasses"
name_key = "neuronNames"

adj = np.array(f[adj_key])
adj[np.arange(len(adj)), np.arange(len(adj))] = 0


def retrieve_field(key):
    references = np.array(f[key][0])
    coded_strings = [f[ref] for ref in references]
    strings = np.array(["".join(chr(c[0]) for c in cl) for cl in coded_strings])
    return strings


labels = retrieve_field(label_key)
names = retrieve_field(name_key)
meta = pd.DataFrame(index=np.arange(len(labels)))
meta["labels"] = labels
meta["names"] = names
meta = meta.set_index("names")

for name in names:
    split_name = name.split("_")
    postfix = split_name[-1]
    if "R" in postfix:
        meta.loc[name, "side"] = "R"
    elif "L" in postfix:
        meta.loc[name, "side"] = "L"
    else:
        meta.loc[name, "side"] = "C"

    postfix = postfix.strip("R")
    postfix = postfix.strip("L")
    meta.loc[name, "designation"] = postfix

meta["inds"] = range(len(meta))

meta = meta.sort_values(["labels", "designation", "side"])
adj = adj[np.ix_(meta["inds"], meta["inds"])]

#%% [markdown]

set_theme()
adjplot(
    adj,
    meta=meta,
    sort_class=["side"],
    item_order=["labels", "designation"],
    plot_type="scattermap",
)

#%%

sym_adj = symmetrize(adj)
undirected_g = nx.from_numpy_array(sym_adj)
str_arange = [f"{i}" for i in range(len(undirected_g))]
arange = np.arange(len(undirected_g))
str_node_map = dict(zip(arange, str_arange))
nx.relabel_nodes(undirected_g, str_node_map, copy=False)
nodelist = meta.index


def optimize_leiden(g, n_restarts=100, resolution=1.0, randomness=0.1):
    best_modularity = -np.inf
    best_partition = {}
    for i in range(n_restarts):
        partition = leiden(
            undirected_g,
            resolution=resolution,
            randomness=randomness,
            check_directed=False,
            extra_forced_iterations=10,
        )
        modularity_score = modularity(
            undirected_g, partitions=partition, resolution=resolution
        )
        if modularity_score > best_modularity:
            best_partition = partition
            best_modularity = modularity_score

    return best_partition, best_modularity


best_partition, mod_score = optimize_leiden(undirected_g, resolution=1.0)
meta["partition"] = list(
    map(best_partition.get, np.arange(len(best_partition)).astype(str))
)

#%%
palette = dict(zip(np.unique(meta["labels"]), cc.glasbey_light))

adjplot(
    pass_to_ranks(adj),
    meta=meta,
    sort_class=["partition", "side"],
    item_order=["side", "labels", "designation"],
    plot_type="heatmap",
    colors=["labels"],
    palette=palette,
    cbar=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),
)
stashfig("adj-modularity")

#%%


def make_palette(cmap="thematic", random_state=None):
    if random_state is None:
        random_state = np.random.default_rng()
    if cmap == "thematic":
        colors = _get_colors(True, None)["nominal"]
    if cmap == "glasbey":
        colors = cc.glasbey_light.copy()
        random_state.shuffle(colors)
    palette = dict(zip(np.unique(labels), colors))
    return palette


seed = 8888888
graphplot(
    adj,
    n_components=32,
    n_neighbors=32,
    embedding_algorithm="ase",
    meta=meta,
    group="partition",
    hue="labels",
    sizes=(20, 90),
    network_order=2,
    normalize_power=True,
    group_convex_hull=True,
    supervised_weight=0.01,
    node_palette=make_palette("thematic"),
    subsample_edges=0.5,
    hue_labels="medioid",
    hue_label_fontsize="xx-small",
    adjust_labels=True,
    random_state=seed,
)
stashfig(f"layout-w-modules-seed={seed}-thematic")

seed = 8888888
graphplot(
    adj,
    n_components=32,
    n_neighbors=32,
    embedding_algorithm="ase",
    meta=meta,
    group="partition",
    hue="labels",
    sizes=(20, 90),
    network_order=2,
    normalize_power=True,
    group_convex_hull=True,
    supervised_weight=0.01,
    node_palette=palette,
    subsample_edges=0.5,
    hue_labels="medioid",
    hue_label_fontsize="xx-small",
    adjust_labels=True,
    random_state=seed,
)
stashfig(f"layout-w-modules-seed={seed}-glasbey")
#%% [markdown]
# ## Generate Layouts
#%%

main_random_state = np.random.default_rng(8888)


def make_legend(palette, ax, s=5):
    elements = []
    legend_labels = []
    for label, color in palette.items():
        element = Line2D(
            [0], [0], marker="o", lw=0, label=label, color=color, markersize=s
        )
        legend_labels.append(label)
        elements.append(element)
    ax.legend(
        handles=elements,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=6,
    )


n_repeats = 0
for i in range(n_repeats):
    random_seed = main_random_state.integers(np.iinfo(np.int32).max)
    for cmap in ["thematic"]:
        random_state = np.random.default_rng(random_seed)
        # random_state = None
        palette = make_palette(cmap, main_random_state)
        ax = graphplot(
            adj,
            n_components=32,
            n_neighbors=32,
            embedding_algorithm="ase",
            meta=meta,
            hue="labels",
            node_spalette=palette,
            sizes=(20, 90),
            network_order=2,
            normalize_power=True,
            random_state=random_state,
            supervised_weight=0.01,
            text_labels=True,
            adjust_labels=True,
        )
        # make_legend(palette, ax)
        stashfig(f"auditory-layout-seed={random_seed}-cmap={cmap}")
#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
make_legend(palette, ax)
ax.axis("off")
stashfig("legend")

#%%
from umap import UMAP
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

n_neighbors = 16
min_dist = 0.2

umapper = UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric="precomputed",
    random_state=random_state.integers(np.iinfo(np.int32).max),
)
# matrix_to_embed = pass_to_ranks(adj)
# shortest_paths = graph_shortest_path(adj, directed=True)
adj_inv = adj.copy()
adj_inv = pass_to_ranks(adj_inv)
adj_inv[adj_inv != 0] = 1 - adj_inv[adj_inv != 0]
shortest_path_dists = shortest_path(csr_matrix(adj_inv), directed=False, unweighted=False)
from graspologic.plot import heatmap

heatmap(shortest_path_dists)

#%%
# for row in shortest_paths:
#     np.argmin(shortest_paths)
# matrix_to_embed = 1 - pass_to_ranks(shortest_paths)
# matrix_to_embed = adj @ adj
X = umapper.fit_transform(shortest_path_dists)
graphplot(
    network=adj,
    embedding=X,
    meta=meta,
    group="partition",
    hue="labels",
    # palette=palette,
    sizes=(20, 90),
    group_convex_hull=False,
    # random_state=random_state,
    node_palette=palette,
    hue_labels="medioid",
    hue_label_fontsize="xx-small",
    random_state=seed,
)

#%%
best_partition, mod_score = optimize_leiden(undirected_g, resolution=1.0)
meta["partition"] = list(
    map(best_partition.get, np.arange(len(best_partition)).astype(str))
)

#%%
from anytree import NodeMixin


class LeidenTree(NodeMixin):
    def __init__(self, min_split=4, children=[], parent=None, verbose=False):
        self.min_split = min_split
        self.children = children
        self.parent = parent
        self.verbose = verbose

    def fit(self, g):
        if self.verbose > 2:
            print(f"Depth: {self.depth}")
            print(f"Number of nodes in subgraph: {len(g)}")
        self.g = g
        self.nodes = sorted(g.nodes)
        self.n_nodes = len(g.nodes)
        if len(g) > self.min_split:
            best_partition, mod_score = optimize_leiden(
                g, resolution=2.0, n_restarts=25
            )

            partition_node_map = {}
            for node in sorted(g.nodes):
                fit_partition = best_partition[node]
                if fit_partition not in partition_node_map:
                    partition_node_map[fit_partition] = []
                partition_node_map[fit_partition].append(node)

            if len(partition_node_map) > 1:
                for partition_key, partition_nodes in partition_node_map.items():
                    sub_g = nx.subgraph(g, partition_nodes).copy()
                    child = LeidenTree(
                        min_split=self.min_split, parent=self, verbose=self.verbose
                    )
                    child.fit(sub_g)


g = undirected_g


lt = LeidenTree(min_split=32, verbose=3)
lt.fit(g)

#%%


#%%
from anytree.walker import Walker

nodelist = sorted(g.nodes)
tree_dists = pd.DataFrame(
    index=nodelist, columns=nodelist, data=np.zeros((len(nodelist), len(nodelist)))
)
w = Walker()
for source_leaf in lt.leaves:
    for target_leaf in lt.leaves:
        up, _, down = w.walk(source_leaf, target_leaf)
        prod = len(up) * len(down)
        if prod > 0:
            distance = np.sqrt(prod)
        else:
            distance = 0
        tree_dists.loc[source_leaf.nodes, target_leaf.nodes] = distance

sns.heatmap(tree_dists.values)

#%%
shortest_paths = graph_shortest_path(sym_adj, directed=True)
shortest_paths = pass_to_ranks(shortest_paths)
shortest_paths = pd.DataFrame(
    index=np.arange(len(adj)), columns=np.arange(len(adj)), data=shortest_paths
)
sns.heatmap(shortest_paths)

#%%
tree_dists.columns = tree_dists.columns.astype(int)
tree_dists.index = tree_dists.index.astype(int)
tree_dists = tree_dists.reindex_like(shortest_paths)

#%%
hybrid_dists = (tree_dists.values + 1) * shortest_paths.values

#%%
umapper.fit_transform(hybrid_dists)
from graspologic.embed import ClassicalMDS

X = ClassicalMDS(n_components=2, dissimilarity="precomputed").fit_transform(
    hybrid_dists
)
graphplot(
    network=adj,
    embedding=X,
    meta=meta,
    # group="partition",
    hue="labels",
    # palette=palette,
    sizes=(20, 90),
    # group_convex_hull=True,
    # random_state=random_state,
    node_palette=palette,
    hue_labels="medioid",
    hue_label_fontsize="xx-small",
    random_state=seed,
)

#%%
from graspologic.plot import heatmap

sns.clustermap(hybrid_dists)


#%% [markdown]
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
