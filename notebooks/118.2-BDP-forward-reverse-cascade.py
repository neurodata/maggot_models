# %% [markdown]
# ## Imports
import os
import time
from itertools import chain

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata

from graspy.cluster import AutoGMMCluster
from src.data import load_metagraph
from src.graph import preprocess
from src.io import savecsv, savefig
from src.traverse import Cascade, TraverseDispatcher, to_transmission_matrix
from src.visualization import CLASS_COLOR_DICT, draw_colors, draw_separators, matrixplot

sns.set_context("talk", font_scale=1.25)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, dpi=200, **kws)
    savefig(name + "high", foldername=FNAME, save_on=True, dpi=400, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


VERSION = "2020-03-26"
print(f"Using version {VERSION}")
# graph_types = ["G", "Gad", "Gaa", "Gdd", "Gda"]
threshold = 0
weight = "weight"

mg = load_metagraph("Gad", VERSION)
mg = preprocess(mg, threshold=0, sym_threshold=False, remove_pdiff=True, binarize=False)
adj = mg.adj
n_verts = len(adj)
meta = mg.meta
meta["inds"] = range(len(meta))

# %% [markdown]
# ## Define cell classes we will be using
out_groups = [
    ("dVNC", "dVNC;CN", "dVNC;RG", "dSEZ;dVNC"),
    ("dSEZ", "dSEZ;CN", "dSEZ;LHN", "dSEZ;dVNC"),
    ("motor-PaN", "motor-MN", "motor-VAN", "motor-AN"),
    ("RG", "RG-IPC", "RG-ITP", "RG-CA-LP", "dVNC;RG"),
    ("dUnk",),
]
out_group_names = ["VNC", "SEZ" "motor", "RG", "dUnk"]
source_groups = [
    ("sens-ORN",),
    ("sens-MN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-thermo",),
    ("sens-vtd",),
    ("sens-AN",),
]
source_group_names = ["Odor", "MN", "Photo", "Temp", "VTD", "AN"]
class_key = "merge_class"


# %% [markdown]
# ## Loop over pairs of in/out classes, run forward/reverse cascade for each


sg = ("sens-ORN",)
# sg = list(chain.from_iterable(source_groups))
og = list(chain.from_iterable(out_groups))
# og = ("dVNC", "dVNC;CN", "dVNC;RG", "dSEZ;dVNC")
sg_name = "Odor"
og_name = "All"

print(f"Running cascades for {sg_name} and {og_name}")


np.random.seed(888)
max_hops = 10
n_init = 100
p = 0.05
traverse = Cascade
simultaneous = True
sorter = "casc_mean_visit"

basename = f"-source={sg_name}-out={og_name}-max_hops={max_hops}-n_init={n_init}-p={p}"

transition_probs = to_transmission_matrix(adj, p)

# %% [markdown]
# ##
currtime = time.time()

source_inds = meta[meta[class_key].isin(sg)]["inds"].values
out_inds = meta[meta[class_key].isin(og)]["inds"].values

td = TraverseDispatcher(
    traverse,
    transition_probs,
    n_init=n_init,
    simultaneous=simultaneous,
    stop_nodes=out_inds,
    max_hops=max_hops,
    allow_loops=False,
)
fwd_hop_hist = td.multistart(source_inds)
fwd_hop_hist = fwd_hop_hist.T

# backward cascade
td = TraverseDispatcher(
    traverse,
    transition_probs.T,
    n_init=n_init,
    simultaneous=simultaneous,
    stop_nodes=source_inds,
    max_hops=max_hops,
    allow_loops=False,
)
back_hop_hist = td.multistart(out_inds)
back_hop_hist = back_hop_hist.T

full_hop_hist = np.concatenate((fwd_hop_hist, back_hop_hist[::-1]), axis=0)
print()

print(f"\n{time.time() - currtime} elapsed\n")
# stashcsv(pd.DataFrame(full_hop_hist), "all_hop_hist")
# path = f"./maggot_models/notebooks/outs/{FNAME}/csvs/all_hop_hist.csv"
# all_hop_df = pd.read_csv(path, index_col=0)
# all_hop_hist = all_hop_df.values

# %% [markdown]
# ##

# add some metadata
hop_range = np.arange(1, max_hops + 1)
hops = np.concatenate((hop_range, -1 * hop_range[::-1]))
hop_order = np.arange(len(hops))
direction = max_hops * [" Fwd"] + max_hops * ["Back"]
hop_data = np.stack((hops, hop_order), axis=1)
row_meta = pd.DataFrame(data=hop_data, columns=["hops", "hop_order"])


def mean_visit(hop_hist):
    n_visits = np.sum(hop_hist, axis=0)
    weight_sum_visits = (hop_range[:, None] * hop_hist).sum(axis=0)
    weight_sum_visits[weight_sum_visits == 0] = (
        max_hops + 1
    )  # cells that are never visited
    mean_visit = weight_sum_visits / n_visits
    return mean_visit


meta["fwd_visit"] = mean_visit(full_hop_hist[:max_hops])
meta["group_fwd_visit"] = meta["merge_class"].map(
    meta.groupby("merge_class")["fwd_visit"].mean()
)

meta["back_visit"] = mean_visit(full_hop_hist[max_hops:][::-1])
meta["group_back_visit"] = meta["merge_class"].map(
    meta.groupby("merge_class")["back_visit"].mean()
)

meta["diff_visit"] = meta["fwd_visit"] - meta["back_visit"]
meta["group_diff_visit"] = meta["merge_class"].map(
    meta.groupby("merge_class")["diff_visit"].mean()
)

colors = sns.color_palette("RdBu_r", n_colors=2 * max_hops)
blues = colors[:max_hops]
reds = colors[max_hops:]
colors = blues[::-1] + reds[::-1]
color_dict = dict(zip(hops, colors))


# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
matrixplot(
    adj,
    ax=ax,
    row_meta=meta,
    row_colors="merge_class",
    row_palette=CLASS_COLOR_DICT,
    row_item_order="diff_visit",
    row_ticks=False,
    col_meta=meta,
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_item_order="diff_visit",
    plot_type="scattermap",
    sizes=(2.5, 5),
)
stashfig(f"adj-diff-sort" + basename)


# %% [markdown]
# ##

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    full_hop_hist,
    ax=ax,
    row_meta=row_meta,
    row_colors="hops",
    row_palette=color_dict,
    row_item_order="hop_order",
    col_meta=meta,
    col_colors="merge_class",
    col_sort_class=["merge_class"],
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_class_order="group_fwd_visit",
    col_item_order="fwd_visit",
)
ax.axhline(max_hops, linewidth=1, linestyle="--", color="grey")
stashfig("full-hop-hist" + basename)

# %% [markdown]
# ##
log_hop_hist = np.log10(full_hop_hist + 1)
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    log_hop_hist,
    ax=ax,
    row_meta=row_meta,
    row_colors=row_meta["hops"].values.astype(int),
    row_palette=color_dict,
    row_item_order="hop_order",
    col_meta=meta,
    col_colors="merge_class",
    col_sort_class="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_class_order="group_fwd_visit",
    col_item_order="fwd_visit",
)
ax.axhline(max_hops, linewidth=1, linestyle="--", color="grey")

stashfig("log-hop-hist" + basename)

# %% [markdown]
# ##


colors = np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"].values)
row_colors = np.vectorize(color_dict.get)(row_meta["hops"].values)
row_colors = np.array(row_colors).T
cg = sns.clustermap(
    log_hop_hist,
    col_cluster=True,
    col_colors=colors,
    row_colors=row_colors,
    cmap="RdBu_r",
    center=0,
    row_cluster=False,
    figsize=(20, 10),
    cbar_pos=None,
    dendrogram_ratio=(0, 0.2),
)
ax = cg.ax_heatmap
ax.axhline(max_hops, linewidth=2, linestyle="--", color="grey")
ax.set_xticks([])
ax.set_yticks([])
cg.ax_row_colors.set_ylabel("Hops")
stashfig("clustermap" + basename)

# %% [markdown]
# ## last thing, gmm


agmm = AutoGMMCluster(
    min_components=10,
    max_components=30,
    affinity=["euclidean", "manhattan"],
    max_agglom_size=3000,
    n_jobs=-2,
    verbose=10,
)

agmm.fit(log_hop_hist.T)
# %% [markdown]
# ##
results = agmm.results_
best_inds = results.groupby("n_components")["bic/aic"].idxmin()
best_results = results.loc[best_inds]
sns.scatterplot(data=best_results, x="n_components", y="bic/aic")
k = 25
best_results = best_results.set_index("n_components")
model = best_results.loc[k, "model"]
pred_labels = model.predict(log_hop_hist.T)

meta["pred_labels"] = pred_labels
meta["pred_fwd_visit"] = meta["pred_labels"].map(
    meta.groupby("pred_labels")["fwd_visit"].mean()
)

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    log_hop_hist,
    ax=ax,
    row_meta=row_meta,
    row_colors="hops",
    row_palette=color_dict,
    row_item_order="hop_order",
    col_meta=meta,
    col_colors="merge_class",
    col_sort_class="pred_labels",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=True,
    col_class_order="pred_fwd_visit",
    col_item_order="diff_visit",
    cbar=True,
)
ax.axhline(max_hops, linewidth=1, linestyle="--", color="grey")

stashfig("gmm-hop-hist" + basename)
