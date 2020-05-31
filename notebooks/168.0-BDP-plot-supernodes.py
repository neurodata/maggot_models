# %% [markdown]
# ##
import os

import numpy as np
import seaborn as sns

from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


mg = load_metagraph("Gs", version="2020-05-26")
adj = mg.adj
meta = mg.meta
meta["inds"] = range(len(meta))

col_meta = meta.copy()
col_meta = col_meta[col_meta["super"]]
col_meta = col_meta.iloc[::-1]
col_inds = col_meta["inds"]

row_meta = meta.copy()
row_meta = row_meta[row_meta["output"]]
row_meta = row_meta.sort_values("merge_class")
row_inds = row_meta["inds"]

incidence = adj[np.ix_(row_inds, col_inds)]

sns.clustermap(incidence, col_cluster=False)

col_colors = col_meta["class2"].map(
    dict(zip(col_meta["class2"].unique(), sns.color_palette("deep")))
)
cg = sns.clustermap(
    pass_to_ranks(incidence),
    col_cluster=False,
    cbar_pos=None,
    yticklabels=False,
    col_colors=col_colors.values,
)
ax = cg.ax_heatmap
ax.set_xticklabels(col_meta["name"], rotation=90)
ax.xaxis.tick_top()
ax.tick_params(axis="both", which="major", pad=25)
stashfig("output-supernode-clustermap")

col_meta["inds"] = range(len(col_meta))
col_meta.index.name = "skid"
col_meta = col_meta.reset_index()
col_meta = col_meta.sort_values(["hemisphere", "skid"], ascending=False)
col_inds = col_meta["inds"]
incidence = incidence[:, col_inds]


col_colors = col_meta["class2"].map(
    dict(zip(col_meta["class2"].unique(), sns.color_palette("deep")))
)
cg = sns.clustermap(
    pass_to_ranks(incidence),
    col_cluster=False,
    cbar_pos=None,
    yticklabels=False,
    col_colors=col_colors.values,
)
ax = cg.ax_heatmap
ax.set_xticklabels(col_meta["name"], rotation=90)
ax.xaxis.tick_top()
ax.tick_params(axis="both", which="major", pad=25)
stashfig("output-supernode-clustermap-sides")
