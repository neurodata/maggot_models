# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import tplquad
from scipy.stats import gaussian_kde

import pymaid
from graspy.utils import pass_to_ranks
from hyppo.ksample import KSample
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig

from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    get_mid_map,
    gridmap,
    matrixplot,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)
from src.spatial import spatial_dcorr, run_dcorr

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# params

level = 7
class_key = f"lvl{level}_labels"

metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "color_iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.2-BDP-omni-clust"


# load data
pair_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
pair_meta["lvl0_labels"] = pair_meta["lvl0_labels"].astype(str)
pair_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
pair_adj = pair_adj.values
mg = MetaGraph(pair_adj, pair_meta)
meta = mg.meta

level_names = [f"lvl{i}_labels" for i in range(level + 1)]


def sort_mg(mg, level_names):
    meta = mg.meta
    sort_class = level_names + ["merge_class"]
    class_order = ["sf"]
    total_sort_by = []
    for sc in sort_class:
        for co in class_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    mg = mg.sort_values(total_sort_by, ascending=False)
    return mg


mg = sort_mg(mg, level_names)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj


# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)


# %% [markdown]
# ##


rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "grey",
    "ytick.color": "dimgrey",
    "xtick.color": "dimgrey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
}
for k, val in rc_dict.items():
    mpl.rcParams[k] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)


def filter_connectors(connectors, ids, direction, compartment):
    label_connectors = connectors[connectors[f"{direction}_to"].isin(ids)]
    label_connectors = label_connectors[
        label_connectors[f"{direction}_type"] == compartment
    ]
    label_connectors = label_connectors[
        ~label_connectors["connector_id"].duplicated(keep="first")
    ]
    return label_connectors


from src.visualization import plot_3view

# start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)

# %% [markdown]
# ##
# compare dendrite inputs


method = "full"

currtime = time.time()

ids = meta.index.values
skid = ids[0]
rows = []
for skid in ids:
    inputs = connectors[connectors[f"postsynaptic_to"] == skid]
    inputs = inputs[~inputs["connector_id"].duplicated(keep="first")]
    outputs = connectors[connectors[f"presynaptic_to"] == skid]
    outputs = outputs[~outputs["connector_id"].duplicated(keep="first")]
    data1 = inputs[["x", "y", "z"]].values
    data2 = outputs[["x", "y", "z"]].values
    stat, p_val = spatial_dcorr(data1, data2)
    row = {"stat": stat, "p_val": p_val, "skid": skid}
    rows.append(row)

results = pd.DataFrame(rows)

print(f"{time.time() - currtime} elapsed")
# %% [markdown]
# ##
results = results.set_index("skid")
meta = pd.concat((meta, results), axis=1)

# %% [markdown]
# ##
var = "stat"
fg = sns.FacetGrid(
    meta,
    col="merge_class",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    col_wrap=10,
    sharex=True,
    sharey=False,
)
fg.map(sns.distplot, var)
stashfig("stat_marginals")

# # %% [markdown]
# # ##
# sort_stat = meta.sort_values(var, ascending=False)
# sort_stat = sort_stat[sort_stat["class1"] != "KC"]
# sort_stat = sort_stat[sort_stat["class1"] != "sens"]

# start_instance()

# skeleton_color_dict = dict(
#     zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
# )
# connection_types = ["axon", "dendrite", "unsplittable"]
# pal = sns.color_palette("deep", 5)
# colors = [pal[1], pal[2], pal[4]]
# connection_colors = dict(zip(connection_types, colors))

# splits = pymaid.find_treenodes(tags="mw axon split")
# splits = splits.set_index("skeleton_id")["treenode_id"].squeeze()


# def plot_fragments(nl, splits, neuron_class=None, scale=8, stat=None):
#     n_col = len(nl)
#     fig = plt.figure(figsize=(scale * n_col, scale))  # constrained_layout=True)
#     for i, n in enumerate(nl):
#         ax = fig.add_subplot(1, n_col, i + 1, projection="3d")
#         skid = int(n.skeleton_id)
#         if skid in splits.index:
#             split_nodes = splits[skid]
#             split_locs = pymaid.get_node_location(split_nodes)
#             split_locs = split_locs[["x", "y", "z"]].values
#             pymaid.plot2d(
#                 split_locs, ax=ax, scatter_kws=dict(color="orchid", s=30), method="3d"
#             )
#             # order of output is axon, dendrite
#             fragments = pymaid.cut_neuron(n, split_nodes)
#         else:
#             fragments = [n]
#         n_frag = len(fragments)
#         for i, f in enumerate(fragments):
#             if n_frag == 1:
#                 color = colors[2]  # unsplitable
#             elif i == n_frag - 1:
#                 color = colors[1]  # dendrite
#             else:
#                 color = colors[0]  # axon
#             f.plot2d(ax=ax, color=color, method="3d")
#             title = f"{neuron_class}, {n.neuron_name}, {n.skeleton_id}, {stat}"
#             ax.set_title(title, color="grey")
#         set_axes_equal(ax)
#     plt.tight_layout()


# def get_savename(nl, neuron_class=None):
#     savename = f"{neuron_class}"
#     for n in nl:
#         savename += f"-{n.skeleton_id}"
#     savename += "-split"
#     return savename


# for skid, row in sort_stat.iloc[:100].iterrows():
#     neuron_class = row["merge_class"]
#     nl = pymaid.get_neurons(skid)
#     nl = pymaid.CatmaidNeuronList(nl)
#     plot_fragments(nl, splits, neuron_class=neuron_class, stat=row[var])
#     stashfig(get_savename(nl, neuron_class=neuron_class))
