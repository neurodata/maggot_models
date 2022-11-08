#%%
import csv
import gzip
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from giskard.plot import set_theme, soft_axis_off
from matplotlib.collections import LineCollection
from src.data import load_maggot_graph, load_palette
from src.io import savefig
from tqdm import tqdm
from src.visualization import HUE_KEY, ORDER_KEY

t0 = time.time()

np.random.seed(8888)

n_components = 10
min_split = 32
level = 7
gap = 50  # 40
width = 0.25
alpha = 0.0075
linewidth = 0.0075
n_subsamples = 100000
# end = "RGN"
GROUP_KEY = f"dc_level_{level}_n_components={n_components}_min_split={min_split}"

set_theme()
palette = load_palette()
mg = load_maggot_graph()

out_path = Path("maggot_models/experiments/plot_paths/")


def stashfig(name, **kws):
    savefig(
        name, pathname=out_path / "figs", format="png", dpi=300, save_on=True, **kws
    )
    savefig(name, pathname=out_path / "figs", format="pdf", save_on=True, **kws)


def open_simple_paths(save_path):
    with gzip.open(save_path, "rt") as f:
        reader = csv.reader(f, delimiter=",")
        recovered_stuff = [list(map(int, row)) for row in reader]
    return recovered_stuff


def sort_meta(meta, sort_class, sort_item=None, class_order=[]):
    meta = meta.copy()
    total_sort_by = []
    for sc in sort_class:
        if len(class_order) > 0:
            for co in class_order:
                class_value = meta.groupby(sc)[co].mean()
                meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
                total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    total_sort_by += sort_item
    meta["sort_idx"] = range(len(meta))
    if len(total_sort_by) > 0:
        meta.sort_values(total_sort_by, inplace=True, kind="mergesort")
    meta["new_idx"] = range(len(meta))
    return meta


def calc_bar_params(sizes, label, top, palette=None):
    heights = sizes.loc[label]
    offset = top
    starts = heights.cumsum() - heights + offset
    colors = np.vectorize(palette.get)(heights.index)
    return heights, starts, colors


paths_loc = (
    f"maggot_models/experiments/plot_paths/data/all_paths_sens-to-{end}_cutoff6.csv.gz"
)

print("Loading paths...")

# get paths
file_name = os.path.basename(paths_loc)
file_name = file_name.split(".")[0]
paths = open_simple_paths(paths_loc)
print(f"Number of paths: {len(paths)}")

# count
max_hops = max(map(len, paths))
print(f"Maximum path length: {max_hops}")

# sort
meta = mg.nodes.copy()
meta = meta[~meta[GROUP_KEY].isna()]
print(f"Number of nodes: {len(meta)}")
meta = sort_meta(meta, [GROUP_KEY, HUE_KEY], [ORDER_KEY], [ORDER_KEY])

# remove paths with references to unclustered nodes
valid_paths = []
for path in paths:
    add_path = True
    for node in path:
        if node not in meta.index:
            add_path = False
            continue
    if add_path:
        valid_paths.append(path)
paths = valid_paths
print(f"Number of paths after removing invalid paths: {len(paths)}")


sizes = meta.groupby([GROUP_KEY, HUE_KEY], sort=False).size()
clusters = sizes.index.unique(level=0)


n_nodes = sizes.sum()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
y_max = n_nodes + gap * len(clusters)
ax.set_xlim((-0.5, max_hops - 0.5))
ax.set_ylim((y_max, 0))

print("Drawing cluster bars...")
# draw cluster bars
mids = []
mid_map = {}
for x in np.arange(max_hops):
    top = 0
    for i, cluster in enumerate(clusters):
        cluster_sizes = sizes.loc[cluster]
        heights, starts, colors = calc_bar_params(sizes, cluster, top, palette=palette)
        if x == 0:
            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            mid = (minimum + maximum) / 2
            mids.append(mid)
            mid_map[cluster] = mid
        for j in range(len(heights)):
            ax.bar(
                x=x,
                height=heights[j],
                width=width,
                bottom=starts[j],
                color=colors[j],
            )
        top += heights.sum() + gap

        if x == 0:
            cluster_idx = meta[meta[GROUP_KEY] == cluster].index
            meta.loc[cluster_idx, "adjusted_idx"] = (
                meta.loc[cluster_idx, "new_idx"] + gap * i
            )


plot_inds = np.random.choice(len(paths), n_subsamples)

print("Creating lines...")
lines = []
for i in tqdm(plot_inds):
    path = paths[i]
    for hop, (source, target) in enumerate(nx.utils.pairwise(path)):
        y1 = meta.loc[source, "adjusted_idx"]
        y2 = meta.loc[target, "adjusted_idx"]
        x1 = hop + width / 2
        x2 = hop + 1 - width / 2
        lines.append([[x1, y1], [x2, y2]])

print("Adding lines...")
lc = LineCollection(lines, colors="black", linewidths=linewidth, alpha=alpha, zorder=-5)
ax.add_collection(lc)

soft_axis_off(ax)
ax.set_xlabel("Hops", fontsize="xx-large")
ax.set_xticks(np.arange(max_hops))
ax.set_xticklabels(np.arange(max_hops) + 1, fontsize="x-large")

print("Saving figure...")
figname = (
    f"paths-plot-n_subsamples={n_subsamples}-file={file_name}"
    f"-linewidth={linewidth}-alpha={alpha}-level={level}"
)
stashfig(figname)
