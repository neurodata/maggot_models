# %% [markdown]
# ##
from src.hierarchy import signal_flow
from src.data import load_metagraph
from src.visualization import matrixplot
from src.visualization import CLASS_COLOR_DICT
from src.io import savefig
import os
from src.graph import preprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain
import networkx as nx


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


VERSION = "2020-03-09"
print(f"Using version {VERSION}")
graph_types = ["G", "Gad", "Gaa", "Gdd", "Gda"]
threshold = 0
weight = "weight"

# load the data
graphs = []
for graph_type in graph_types:
    mg = load_metagraph(graph_type, VERSION)
    mg = preprocess(
        mg,
        threshold=threshold,
        sym_threshold=False,
        remove_pdiff=True,
        binarize=False,
        weight=weight,
    )
    print(
        f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}"
    )
    graphs.append(mg)

inout = "sensory_to_out"
if inout == "sensory_to_out":
    out_classes = [
        "O_dSEZ",
        "O_dSEZ;CN",
        "O_dSEZ;LHN",
        "O_dVNC",
        "O_dVNC;O_RG",
        "O_dVNC;CN",
        "O_RG",
        "O_dUnk",
        "O_RG-IPC",
        "O_RG-ITP",
        "O_RG-CA-LP",
    ]
    from_groups = [
        ("sens-ORN",),
        ("sens-photoRh5", "sens-photoRh6"),
        ("sens-MN",),
        ("sens-thermo",),
        ("sens-vtd",),
        ("sens-AN",),
    ]
    from_group_names = ["Odor", "Photo", "MN", "Temp", "VTD", "AN"]

if inout == "out_to_sensory":
    from_groups = [
        ("motor-mAN", "motormVAN", "motor-mPaN"),
        ("O_dSEZ", "O_dVNC;O_dSEZ", "O_dSEZ;CN", "LHN;O_dSEZ"),
        ("O_dVNC", "O_dVNC;CN", "O_RG;O_dVNC", "O_dVNC;O_dSEZ"),
        ("O_RG", "O_RG-IPC", "O_RG-ITP", "O_RG-CA-LP", "O_RG;O_dVNC"),
        ("O_dUnk",),
    ]
    from_group_names = ["Motor", "SEZ", "VNC", "RG", "dUnk"]
    out_classes = [
        "sens-ORN",
        "sens-photoRh5",
        "sens-photoRh6",
        "sens-MN",
        "sens-thermo",
        "sens-vtd",
        "sens-AN",
    ]

from_classes = list(chain.from_iterable(from_groups))  # make this a flat list

class_key = "Merge Class"


# %% [markdown]
# ## signal flow sort and plot


def compute_mean_visit(hop_hist):
    n_visits = np.sum(hop_hist, axis=0)
    weight_sum_visits = (np.arange(1, max_hops + 1)[:, None] * hop_hist).sum(axis=0)
    mean_visit = weight_sum_visits / n_visits
    return mean_visit


from src.traverse import to_transmission_matrix, Cascade, TraverseDispatcher

sns.set_context("talk", font_scale=1.25)

max_hops = 10
n_init = 10
p = 0.03
traverse = Cascade
simultaneous = True

graph_sfs = []
print("Generating cascades")
for mg, graph_type in zip(graphs, graph_types):
    print(graph_type)
    # get out the graph and relevant nodes
    adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
    n_verts = len(adj)
    meta = mg.meta
    meta["inds"] = range(len(meta))
    from_inds = meta[meta[class_key].isin(from_classes)]["inds"].values
    out_inds = meta[meta[class_key].isin(out_classes)]["inds"].values

    # forward cascade
    transition_probs = to_transmission_matrix(adj, p)

    td = TraverseDispatcher(
        traverse,
        transition_probs,
        n_init=n_init,
        simultaneous=simultaneous,
        stop_nodes=out_inds,
        max_hops=max_hops,
        allow_loops=False,
    )
    casc_hop_hist = td.multistart(from_inds)
    casc_hop_hist = casc_hop_hist.T

    # backward cascade
    td = TraverseDispatcher(
        traverse,
        transition_probs.T,
        n_init=n_init,
        simultaneous=simultaneous,
        stop_nodes=from_inds,
        max_hops=max_hops,
        allow_loops=False,
    )
    back_hop_hist = td.multistart(out_inds)
    back_hop_hist = back_hop_hist.T

    meta["casc_mean_visit"] = compute_mean_visit(casc_hop_hist)
    meta["back_mean_visit"] = compute_mean_visit(back_hop_hist)
    meta["diff"] = meta["casc_mean_visit"] - meta["back_mean_visit"]

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    matrixplot(
        mg.adj,
        ax=ax,
        col_meta=meta,
        row_meta=meta,
        col_item_order="diff",
        row_item_order="diff",
        col_colors="Merge Class",
        row_colors="Merge Class",
        col_palette=CLASS_COLOR_DICT,
        row_palette=CLASS_COLOR_DICT,
        plot_type="scattermap",
        sizes=(2.5, 5),
    )
    fig.suptitle(f"{graph_type}, signal flow sorted", y=0.91)
    stashfig(f"sort-scattermap-{graph_type}")

# %% [markdown]
# ## plot the rank orders for each

from scipy.stats import rankdata

sfs = []
rank_sfs = []
for mg, name in zip(graphs, graph_types):
    sf = mg.meta["signal_flow"].copy()
    sf.name = name
    sfs.append(sf)
    rank_sf = rankdata(sf)
    rank_sf = pd.Series(index=sf.index, data=rank_sf, name=name)
    rank_sfs.append(rank_sf)

sf_df = pd.DataFrame(sfs).T
sns.pairplot(sf_df)
# %% [markdown]
# ##
rank_sf_df = pd.DataFrame(rank_sfs).T
rank_sf_df.loc[meta.index, "class"] = meta["Merge Class"]
pg = sns.PairGrid(
    rank_sf_df, vars=graph_types, hue="class", palette=CLASS_COLOR_DICT, corner=True
)
pg.map_offdiag(sns.scatterplot, s=5, alpha=0.5, linewidth=0)


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


def remove_diag(x, **kws):
    ax = plt.gca()
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


pg.map_offdiag(tweak)
pg.map_diag(remove_diag)
stashfig("rank-sf-pairs")
# %% [markdown]
# ## plot sorted by some kind of random walk thingy

