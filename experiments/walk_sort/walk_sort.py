#%%

from itertools import chain
from pathlib import Path
from giskard import graph

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
from src.data import load_maggot_graph, join_node_meta
from src.io import savecsv, savefig
from src.visualization import CLASS_COLOR_DICT, set_theme

set_theme()

# meta_loc = f"{DATA_DIR}/{DATA_VERSION}/meta_data.csv"

mg = load_maggot_graph()
mg.to_largest_connected_component()
meta = mg.nodes
adj = mg.ad.adj
meta["degree"] = np.sum(adj, axis=0) + np.sum(adj, axis=1)
# meta = pd.read_csv(meta_loc, index_col=0)

save_path = Path("maggot_models/experiments/walk_sort/")


def stashfig(name, **kws):
    savefig(name, pathname=save_path / "figs", format="pdf", save_on=True, **kws)
    savefig(name, pathname=save_path / "figs", format="png", save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, pathname=save_path / "outs", **kws)


# %%

graph_type = "sum"
n_init = 256
max_hops = 16
allow_loops = False
include_reverse = True
walk_path = "maggot_models/experiments/walk_sort/outs/walks-"
walk_spec = f"gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}"
forward_walk_path = walk_path + walk_spec + "-reverse=False" + ".txt"
backward_walk_path = walk_path + walk_spec + "-reverse=True" + ".txt"

np.random.seed(8888)


def process_paths(walk_path):
    with open(walk_path, "r") as f:
        paths = f.read().splitlines()
    print(f"# of paths: {len(paths)}")

    paths = list(set(paths))
    paths.remove("")
    print(f"# of paths after removing duplicates: {len(paths)}")

    # n_subsample = len(paths)  # 2 ** 14
    # choice_inds = np.random.choice(len(paths), n_subsample, replace=False)
    # new_paths = []
    # for i in range(len(paths)):
    #     if i in choice_inds:
    #         new_paths.append(paths[i])
    # paths = new_paths

    # print(f"# of paths after subsampling: {len(paths)}")
    paths = [path.split(" ") for path in paths]
    paths = [[int(node) for node in path] for path in paths]
    # all_nodes = set()
    # [[all_nodes.add(node) for node in path] for path in paths]
    # uni_nodes = np.unique(list(all_nodes))
    # ind_map = dict(zip(uni_nodes, range(len(uni_nodes))))
    # tokenized_paths = [list(map(ind_map.get, path)) for path in paths]
    return paths


forward_paths = process_paths(forward_walk_path)
backward_paths = process_paths(backward_walk_path)


# %%
all_nodes = set()
node_visits = {}
for path in forward_paths:
    for i, node in enumerate(path):
        if node not in node_visits:
            node_visits[node] = []
        node_visits[node].append(i / (len(path) - 1))
[[all_nodes.add(node) for node in path] for path in forward_paths]

if include_reverse:
    for path in backward_paths:
        for i, node in enumerate(path):
            if node not in node_visits:
                node_visits[node] = []
            node_visits[node].append(1 - (i / (len(path) - 1)))
    [[all_nodes.add(node) for node in path] for path in backward_paths]

uni_nodes = np.unique(list(all_nodes))

median_node_visits = {}
for node in uni_nodes:
    median_node_visits[node] = np.median(node_visits[node])
meta["median_node_visits"] = meta.index.map(median_node_visits)
visits = meta["median_node_visits"]
visits.name = f"{graph_type}_walk_sort"
join_node_meta(visits, overwrite=True)

median_class_visits = {}
for node_class in meta["merge_class"].unique():
    nodes = meta[meta["merge_class"] == node_class].index
    all_visits_in_class = list(map(node_visits.get, nodes))
    all_visits_in_class = [item for item in all_visits_in_class if item is not None]
    all_visits_flat = list(chain.from_iterable(all_visits_in_class))
    median_class_visits[node_class] = np.median(all_visits_flat)
meta["median_class_visits"] = meta["merge_class"].map(median_class_visits)

meta.to_csv(
    f"maggot_models/experiments/walk_sort/outs/meta_w_order-{walk_spec}-include_reverse={include_reverse}.csv"
)

print(f"# of nodes: {len(meta)}")
unvisit_meta = meta[meta["median_node_visits"].isna()]
print(f"# of unvisited nodes: {len(unvisit_meta)}")
# relevant_cols = [
#     "brain_neurons",
#     "left",
#     "right",
#     "sink",
#     "preliminary_LN",
#     "partially_differentiated",
#     "unsplittable",
#     "output",
#     "input",
#     "class1",
#     "all_class1",
#     "n_class1",
#     "class2",
#     "all_class2",
#     "n_class2",
#     "simple_class",
#     "all_simple_class",
#     "n_simple_class",
#     "hemisphere",
#     "pair",
#     "pair_id",
#     "merge_class",
#     "lineage",
#     "dendrite_input",
#     "axon_input",
#     "name",
#     "median_node_visits",
#     "median_class_visits",
# ]
# unvisit_meta = unvisit_meta[relevant_cols]
unvisit_meta.sort_values(
    ["class1", "class2", "merge_class", "pair_id", "hemisphere"], inplace=True
)
unvisit_meta.to_csv(
    f"maggot_models/experiments/walk_sort/outs/unvisited_meta-{walk_spec}.csv"
)


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
stashfig(f"class-rw-order-heatmap-{walk_spec}")


# %%

fig, axs = plt.subplots(
    1,
    2,
    figsize=(10, 20),
    gridspec_kw=dict(width_ratios=[0.05, 0.95], wspace=0),
    sharey=True,
)
ax.set(ylim=(0, len(meta)), xlim=(-0.02, 1.02))

ax = axs[0]
sort_by = ["median_node_visits"]
sort_meta = meta.sort_values(sort_by)
sort_meta["ind"] = range(len(sort_meta))
color_dict = CLASS_COLOR_DICT
classes = sort_meta["merge_class"].values
uni_classes = np.unique(sort_meta["merge_class"])
class_map = dict(zip(uni_classes, range(len(uni_classes))))
color_sorted = np.vectorize(color_dict.get)(uni_classes)
lc = ListedColormap(color_sorted)
class_indicator = np.vectorize(class_map.get)(classes)
class_indicator = class_indicator.reshape(len(classes), 1)

sns.heatmap(
    class_indicator,
    cmap=lc,
    cbar=False,
    yticklabels=False,
    xticklabels=False,
    square=False,
    ax=ax,
)
ax.set(ylabel="Neurons")
# ax.axis("off")

ax = axs[1]
ax.set(
    ylim=(len(meta), 0),
    xlim=(-0.02, 1.02),
    xlabel="Normalized hop time",
    title=f"Sort by = {sort_by}",
)

for i, (node_id, row) in enumerate(sort_meta.iterrows()):
    if node_id in node_visits:
        visits = node_visits[node_id]
        ax.plot(
            visits,
            np.array(len(visits) * [i]),
            color="black",
            alpha=0.2,
            marker="|",
            markersize=1,
            # s=2,
            linewidth=0,
            # linewidths=0.1,
        )
        median = row["median_node_visits"]
        ax.plot(
            median,
            i,
            color="red",
            alpha=1,
            markersize=2.5,
            marker="|",
            zorder=99,
        )


custom_lines = [
    Line2D([0], [0], color="black", marker="|", lw=0, markersize=10),
    Line2D([0], [0], color="red", marker="|", lw=0, markersize=10),
]

ax.legend(
    custom_lines,
    ["All visits", "Median"],
    bbox_to_anchor=(1, 1),
    loc="upper right",
)


stashfig(f"hop-time-plot-sort_by={sort_by}-{walk_spec}")
