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

from src.data import load_metagraph
from src.graph import preprocess
from src.io import savecsv, savefig
from src.traverse import Cascade, TraverseDispatcher, to_transmission_matrix
from src.visualization import CLASS_COLOR_DICT, matrixplot

sns.set_context("talk", font_scale=1.25)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


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
    # ("dSEZ", "dSEZ;CN", "dSEZ;LHN", "dSEZ;dVNC"),
    # ("motor-PaN", "motor-MN", "motor-VAN", "motor-AN"),
    # ("RG", "RG-IPC", "RG-ITP", "RG-CA-LP", "dVNC;RG"),
    # ("dUnk",),
]
out_group_names = ["VNC", "SEZ"]  # "motor", #"RG", "dUnk"]
source_groups = [
    ("sens-ORN",),
    # ("sens-MN",),
    # ("sens-photoRh5", "sens-photoRh6"),
    # ("sens-thermo",),
    # ("sens-vtd",),
    # ("sens-AN",),
]
source_group_names = ["Odor", "MN", "Photo", "Temp", "VTD", "AN"]
class_key = "merge_class"


# %% [markdown]
# ## Loop over pairs of in/out classes, run forward/reverse cascade for each

currtime = time.time()

np.random.seed(888)
max_hops = 10
n_init = 100
p = 0.05
traverse = Cascade
simultaneous = True
sorter = "casc_mean_visit"
graph_sfs = []

transition_probs = to_transmission_matrix(adj, p)
#%%


pair_hop_hists = []
for sg, sg_name in zip(source_groups, source_group_names):
    for og, og_name in zip(out_groups, out_group_names):
        print(f"Running cascades for {sg_name} and {og_name}")

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
        pair_hop_hists.append(full_hop_hist)
        print()

all_hop_hist = np.concatenate(pair_hop_hists, axis=0)

print(f"\n{time.time() - currtime} elapsed\n")
stashcsv(pd.DataFrame(all_hop_hist), "all_hop_hist")

# %% [markdown]
# ##
path = f"./maggot_models/notebooks/outs/{FNAME}/csvs/all_hop_hist.csv"
all_hop_df = pd.read_csv(path, index_col=0)
all_hop_hist = all_hop_df.values

# %% [markdown]
# ##
source = []
out = []
for sg, sg_name in zip(source_groups, source_group_names):
    for og, og_name in zip(out_groups, out_group_names):
        source.append(sg_name)
        out.append(og_name)

source_indicator = np.repeat(source, 2 * max_hops)
out_indicator = np.repeat(out, 2 * max_hops)

hops = np.concatenate(
    (np.arange(1, max_hops + 1), -1 * np.arange(1, max_hops + 1)[::-1])
)
hop_indicator = np.tile(hops, (len(source_groups) * len(out_groups)))

hop_order = np.arange(len(hop_indicator))
hop_data = np.stack((hop_indicator, hop_order), axis=1)
row_meta = pd.DataFrame(data=hop_data, columns=["hops", "hop_order"])


# hop_hist = all_hop_hist
# n_visits = np.sum(hop_hist, axis=0)
# weight_sum_visits = (hop_indicator[:, None] * hop_hist).sum(axis=0)
# weight_sum_visits[weight_sum_visits == 0] = np.inf  # cells that are never visited
# mean_visit = weight_sum_visits / n_visits


colors = sns.color_palette("RdBu_r", n_colors=2 * max_hops)
blues = colors[:max_hops]
reds = colors[max_hops:]
colors = blues[::-1] + reds
color_dict = dict(zip(hops, colors))

# meta["mean_visit"] = mean_visit
# mapper = meta.groupby("merge_class")["mean_visit"].mean()
# meta["group_mean_visit"] = meta["merge_class"].map(mapper)

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
matrixplot(
    adj,
    ax=ax,
    row_meta=meta,
    row_colors="merge_class",
    row_palette=CLASS_COLOR_DICT,
    row_item_order="mean_visit",
    row_ticks=False,
    col_meta=meta,
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_item_order="mean_visit",
    plot_type="scattermap",
    sizes=(2.5, 5),
)
stashfig("fwd-reverse-diff-sort")


# %% [markdown]
# ##

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    all_hop_hist,
    ax=ax,
    row_meta=row_meta,
    row_colors=row_meta["hops"].values.astype(int),
    row_palette=color_dict,
    row_item_order="hop_order",
    # row_sort_class=["pair"],
    col_meta=meta,
    col_colors="merge_class",
    col_sort_class=["merge_class"],
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_class_order="group_mean_visit",
    col_item_order="mean_visit",
)
stashfig("all_hop_hist")

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    np.log10(all_hop_hist + 1),
    ax=ax,
    row_meta=row_meta,
    row_colors=row_meta["hops"].values.astype(int),
    row_palette=color_dict,
    row_item_order="hop_order",
    col_meta=meta,
    col_colors="merge_class",
    col_sort_class=["merge_class"],
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_class_order="group_mean_visit",
    col_item_order="mean_visit",
)
stashfig("log_hop_hist")

# %% [markdown]
# ##
from src.visualization import draw_separators, draw_colors


colors = np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"].values)
row_colors = np.vectorize(color_dict.get)(row_meta["hops"].values)
row_colors = np.array(row_colors).T
cg = sns.clustermap(
    np.log10(all_hop_hist + 1),
    col_cluster=True,
    col_colors=colors,
    row_colors=row_colors,
    cmap="RdBu_r",
    center=0,
    row_cluster=False,
    figsize=(25, 10),
    cbar_pos=None,
    dendrogram_ratio=(0, 0.2),
)
ax = cg.ax_heatmap
ax.axhline(10, linewidth=2, linestyle="--", color="grey")
ax.set_xticks([])
ax.set_yticks([])
cg.ax_row_colors.set_ylabel("Hops")
# draw_separators(ax, ax_type="y", sort_meta=row_meta, sort_class=)
# draw_colors(ax, ax_type="y", sort_meta=row_meta, colors="hops", palette=color_dict)
stashfig("clustermap-odor-vnc")

# %% [markdown]
# ## Scree plots

from src.visualization import screeplot

screeplot(all_hop_hist, show_first=40)
stashfig("scree-first-40")
screeplot(all_hop_hist, show_first=None)
stashfig("scree-all")
screeplot(np.log10(all_hop_hist + 1), show_first=100)
screeplot(np.log10(all_hop_hist + 1), show_first=100, cumulative=True)

# %% [markdown]
# ##

from graspy.cluster import AutoGMMCluster

agmm = AutoGMMCluster(
    min_components=2,
    max_components=50,
    affinity=["euclidean", "manhattan"],
    max_agglom_size=3000,
    n_jobs=-2,
    verbose=10,
)
agmm.fit(all_hop_hist.T)

# %% [markdown]
# ##

from graspy.embed import select_dimension

select_dimension(all_hop_hist.T, n_elbows=5)
#%%
from graspy.embed import selectSVD
from graspy.plot import pairplot

n_elbows = 3
U, S, V = selectSVD(all_hop_hist.T, n_elbows=n_elbows)

plot_df = pd.DataFrame(data=U)
plot_df["label"] = meta["merge_class"].values

pg = sns.PairGrid(
    plot_df, hue="label", palette=CLASS_COLOR_DICT, vars=np.arange(U.shape[1]), height=4
)

# pg._legend.remove()
# pg.map_diag(plt.hist)
pg.map_offdiag(sns.scatterplot, s=15, linewidth=0, alpha=0.7)


def tweak(x, y, **kws):
    ax = plt.gca()
    if len(x) > 0:
        xmax = np.nanmax(x)
        xtop = ax.get_xlim()[-1]
        if xmax > xtop:
            ax.set_xlim([-1, xmax + 1])
    if len(y) > 0:
        ymax = np.nanmax(y)
        ytop = ax.get_ylim()[-1]
        if ymax > ytop:
            ax.set_ylim([-1, ymax + 1])
    ax.set_xticks([])
    ax.set_yticks([])


pg.map_offdiag(tweak)
stashfig(f"pairs-all-hop-hist-elbows={n_elbows}", dpi=300)
plt.close()
# %% [markdown]
# ##

from graspy.cluster import AutoGMMCluster

agmm = AutoGMMCluster(
    min_components=2,
    max_components=50,
    affinity=["euclidean", "manhattan"],
    max_agglom_size=3000,
    n_jobs=-2,
    verbose=10,
)
agmm.fit(U)

# %% [markdown]
# ##

best_results = agmm.results_.groupby("n_components")["bic/aic"].min()
best_results = best_results.reset_index()
sns.scatterplot(data=best_results, x="n_components", y="bic/aic")

# %% [markdown]
# ##
import colorcet as cc

pred_labels = agmm.predict(U)


plot_df = pd.DataFrame(data=U)
plot_df["label"] = pred_labels

pg = sns.PairGrid(
    plot_df, hue="label", palette=cc.glasbey_light, vars=np.arange(U.shape[1]), height=8
)

# pg._legend.remove()
# pg.map_diag(plt.hist)
pg.map_offdiag(sns.scatterplot, s=15, linewidth=0, alpha=0.7)


def tweak(x, y, **kws):
    ax = plt.gca()
    if len(x) > 0:
        xmax = np.nanmax(x)
        xtop = ax.get_xlim()[-1]
        if xmax > xtop:
            ax.set_xlim([-1, xmax + 1])
    if len(y) > 0:
        ymax = np.nanmax(y)
        ytop = ax.get_ylim()[-1]
        if ymax > ytop:
            ax.set_ylim([-1, ymax + 1])
    ax.set_xticks([])
    ax.set_yticks([])


pg.map_offdiag(tweak)
stashfig("pairs-all-hop-hist-pred", dpi=300)
plt.close()
# %% [markdown]
# ##

from src.visualization import barplot_text

barplot_text(pred_labels, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT)
stashfig("agmm-barplot")
# %% [markdown]
# ## Per known class, plot a rugplot of fwd visit, backwards visit, for odor to
# one of the backwards channels

# %% [markdown]
# ## swarmplots, color by class


# %% [markdown]
# ## Look at some of the individual pathways

source_inds = row_meta[row_meta["source"] == "Odor"].index
mini_hop_hist = hop_hist[source_inds, :]
mini_row_meta = row_meta.iloc[source_inds].copy()

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    np.log10(mini_hop_hist + 1),
    ax=ax,
    row_meta=mini_row_meta,
    row_colors=mini_row_meta["hops"].values.astype(int),
    row_palette=color_dict,
    row_item_order="hop_order",
    row_sort_class=["pair"],
    col_meta=meta,
    col_colors="merge_class",
    col_sort_class=["merge_class"],
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    col_class_order="group_mean_visit",
    col_item_order="mean_visit",
)
stashfig("log_hop_hist_odor")

# %% [markdown]
# ##
agmm = AutoGMMCluster(
    min_components=10,
    max_components=40,
    affinity=["euclidean", "manhattan"],
    max_agglom_size=3000,
    n_jobs=-2,
    verbose=10,
)

agmm.fit(mini_hop_hist.T)

# %% [markdown]
# ##
sg_name = "Odor"
og_name = "VNC"
sg = source_groups[0]
print(sg)
og = out_groups[0]
print(og)
print(f"Running cascades for {sg_name} and {og_name}")

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

full_hop_hist = np.concatenate((fwd_hop_hist, back_hop_hist[::-1, :]), axis=0)
# pair_hop_hists.append(full_hop_hist)
print()

# %% [markdown]
# ##
hop_range = np.arange(1, max_hops + 1)
hops = np.concatenate((hop_range, -1 * hop_range[::-1]))
# hop_indicator = np.tile(hops, (len(source_groups) * len(out_groups)))
# hop_order = np.tile(np.arange(max_hops * 2), len(source_groups) * len(out_groups))
hop_order = np.arange(len(hop_indicator))

hop_data = np.stack((hop_indicator, hop_order, source_indicator, out_indicator), axis=1)
row_meta = pd.DataFrame(data=hop_data, columns=["hops", "hop_order", "source", "out"])

row_meta["pair"] = row_meta["source"] + "-" + row_meta["out"]

# %% [markdown]
# ##
odor_meta = meta.copy()

hop_indicator = np.arange(1, max_hops + 1)
hop_hist = fwd_hop_hist
n_visits = np.sum(hop_hist, axis=0)
weight_sum_visits = (hop_indicator[:, None] * hop_hist).sum(axis=0)
weight_sum_visits[weight_sum_visits == 0] = 0  # cells that are never visited
mean_visit = weight_sum_visits / n_visits
odor_meta["fwd_mean_visit"] = mean_visit

hop_hist = back_hop_hist
n_visits = np.sum(hop_hist, axis=0)
weight_sum_visits = (hop_indicator[:, None] * hop_hist).sum(axis=0)
weight_sum_visits[weight_sum_visits == 0] = 0  # cells that are never visited
mean_visit = weight_sum_visits / n_visits
odor_meta["back_mean_visit"] = mean_visit

odor_meta["diff_visit"] = odor_meta["fwd_mean_visit"] - odor_meta["back_mean_visit"]

mean_series = odor_meta.groupby("merge_class")["diff_visit"].mean()
odor_meta["group_diff_visit"] = odor_meta["merge_class"].map(mean_series)

mean_series = odor_meta.groupby("merge_class")["fwd_mean_visit"].mean()
odor_meta["group_fwd_visit"] = odor_meta["merge_class"].map(mean_series)


fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    np.log10(full_hop_hist + 1),
    ax=ax,
    col_meta=odor_meta,
    col_colors="merge_class",
    col_sort_class=["merge_class"],
    col_class_order="group_diff_visit",
    col_item_order="diff_visit",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
)
stashfig("log_hop_hist_odor")

# %% [markdown]
# ##

odor_meta.sort_values("group_diff_visit")

# %% [markdown]
# ##

agmm = AutoGMMCluster(
    min_components=20,
    max_components=60,
    affinity=["euclidean", "manhattan"],
    max_agglom_size=3000,
    n_jobs=-2,
    verbose=10,
)

agmm.fit(np.log10(full_hop_hist + 1).T)
# %% [markdown]
# ##

results = agmm.results_
best_inds = results.groupby("n_components")["bic/aic"].idxmin()
best_results = results.loc[best_inds]
sns.scatterplot(data=best_results, x="n_components", y="bic/aic")

# %% [markdown]
# ##
model = best_results.loc[213, "model"]

pred_labels = model.predict(np.log10(full_hop_hist + 1).T)
odor_meta["pred_labels"] = pred_labels
mean_series = odor_meta.groupby("pred_labels")["diff_visit"].mean()
odor_meta["pred_diff_visit"] = odor_meta["pred_labels"].map(mean_series)

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
matrixplot(
    np.log10(full_hop_hist + 1),
    ax=ax,
    col_meta=odor_meta,
    col_colors="merge_class",
    col_sort_class=["pred_labels"],
    col_class_order="pred_diff_visit",
    col_item_order="diff_visit",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=True,
    tick_rot=90,
)
stashfig("log_hop_hist_odor_pred")


# %%
agmm2 = AutoGMMCluster(
    min_components=55,
    max_components=70,
    affinity=["euclidean", "manhattan"],
    max_agglom_size=3000,
    n_jobs=-2,
    verbose=10,
)

agmm2.fit(np.log10(full_hop_hist + 1).T)

# %% [markdown]
# ##
results = agmm2.results_
best_inds = results.groupby("n_components")["bic/aic"].idxmin()
best_results = results.loc[best_inds]
sns.scatterplot(data=best_results, x="n_components", y="bic/aic")
