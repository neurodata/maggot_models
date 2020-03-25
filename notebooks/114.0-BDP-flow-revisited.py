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


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


VERSION = "2020-03-09"
print(f"Using version {VERSION}")
graph_types = ["G", "Gad", "Gaa", "Gdd", "Gda"]

# load the data
graphs = []
for graph_type in graph_types:
    threshold = 0
    weight = "weight"
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
# %% [markdown]
# ## signal flow sort and plot
sns.set_context("talk", font_scale=1.25)

graph_sfs = []
for mg, graph_type in zip(graphs, graph_types):
    meta = mg.meta
    sf = signal_flow(mg.adj)
    meta["signal_flow"] = -sf
    graph_sfs.append(sf)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    matrixplot(
        mg.adj,
        ax=ax,
        col_meta=meta,
        row_meta=meta,
        col_item_order="signal_flow",
        row_item_order="signal_flow",
        col_colors="Merge Class",
        row_colors="Merge Class",
        col_palette=CLASS_COLOR_DICT,
        row_palette=CLASS_COLOR_DICT,
        plot_type="scattermap",
        sizes=(2.5, 5),
    )
    fig.suptitle(f"{graph_type}, signal flow sorted", y=0.91)
    stashfig(f"sf-sort-scattermap-{graph_type}")

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

