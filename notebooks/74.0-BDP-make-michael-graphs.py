# %% [markdown]
# #
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_metagraph
from src.io import savefig
from src.visualization import gridmap

SAVEFIGS = True
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, **kws)


brain_version = "2020-01-29"

graph_versions = ["G", "Gad", "Gaa", "Gdd", "Gda", "Gn", "Gadn", "Gaan", "Gddn", "Gdan"]

for graph_version in graph_versions:
    # sort the graph
    mg = load_metagraph(graph_version, brain_version)
    paired_inds = np.where(mg.meta["Pair ID"] != -1)[0]
    mg = mg.reindex(paired_inds)
    mg.sort_values(["Merge Class", "Pair ID", "Hemisphere"], ascending=True)
    if graph_version not in ["G", "Gn"]:
        mg.verify(n_checks=10000, graph_type=graph_version)

    # plot the sorted graph
    mg.meta["Index"] = range(len(mg))
    groups = mg.meta.groupby("Merge Class", as_index=True)
    tick_locs = groups["Index"].mean()
    border_locs = groups["Index"].first()

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    gridmap(mg.adj, sizes=(3, 5), ax=ax)
    for bl in border_locs:
        ax.axvline(bl, linewidth=1, linestyle="--", color="grey", alpha=0.5)
        ax.axhline(bl, linewidth=1, linestyle="--", color="grey", alpha=0.5)

    ticklabels = np.array(list(groups.groups.keys()))
    for axis in [ax.yaxis, ax.xaxis]:
        axis.set_major_locator(plt.FixedLocator(tick_locs[0::2]))
        axis.set_minor_locator(plt.FixedLocator(tick_locs[1::2]))
        axis.set_minor_formatter(plt.FormatStrFormatter("%s"))
    ax.tick_params(which="minor", pad=80)
    ax.set_yticklabels(ticklabels[0::2])
    ax.set_yticklabels(ticklabels[1::2], minor=True)
    ax.set_xticklabels(ticklabels[0::2])
    ax.set_xticklabels(ticklabels[1::2], minor=True)
    ax.xaxis.tick_top()
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    for tick in ax.get_xticklabels(minor=True):
        tick.set_rotation(90)
    stashfig(f"sorted-adj-{graph_version}")

    # save
    out_path = Path(f"./maggot_models/data/interim/{brain_version}")
    adj_df = pd.DataFrame(data=mg.adj, index=mg.meta.index, columns=mg.meta.index)
    adj_df.to_csv(out_path / f"{graph_version}-pair-sorted.csv")

