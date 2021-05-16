#%% [markdown]
# # Layouts for an auditory connectome
#%%
import datetime
import time
from pathlib import Path
from adjustText import adjust_text
import matplotlib.pyplot as plt
import colorcet as cc
import h5py
import numpy as np
import pandas as pd
from giskard.plot import graphplot
from graspologic.layouts.colors import _get_colors
from matplotlib.lines import Line2D
from src.io import savefig
from src.visualization.settings import set_theme

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
from src.visualization import adjplot, set_theme

set_theme()
adjplot(
    adj,
    meta=meta,
    sort_class=["side"],
    item_order=["labels", "designation"],
    plot_type="scattermap",
)

#%%
from graspologic.utils import symmetrize
from graspologic.partition import leiden, modularity
import networkx as nx

sym_adj = symmetrize(adj)
undirected_g = nx.from_numpy_array(sym_adj)
str_arange = [f"{i}" for i in range(len(undirected_g))]
arange = np.arange(len(undirected_g))
str_node_map = dict(zip(arange, str_arange))
nx.relabel_nodes(undirected_g, str_node_map, copy=False)
nodelist = meta.index


def optimize_leiden(g, n_restarts=100, resolution=1.0, randomoness=0.001):
    best_modularity = -np.inf
    best_partition = {}
    for i in range(n_restarts):
        partition = leiden(
            undirected_g,
            resolution=resolution,
            randomness=0.1,
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


best_partition, mod_score = optimize_leiden(undirected_g)
meta["partition"] = list(
    map(best_partition.get, np.arange(len(best_partition)).astype(str))
)

#%%
palette = dict(zip(np.unique(meta["labels"]), cc.glasbey_light))
from graspologic.utils import pass_to_ranks

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
    # palette=palette,
    sizes=(20, 90),
    network_order=2,
    normalize_power=True,
    group_convex_hull=True,
    # random_state=random_state,
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
    # palette=palette,
    sizes=(20, 90),
    network_order=2,
    normalize_power=True,
    group_convex_hull=True,
    # random_state=random_state,
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


n_repeats = 5
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
            palette=palette,
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

#%% [markdown]
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
