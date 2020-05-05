# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from joblib import Parallel, delayed
import pandas as pd

from graspy.match import GraphMatch
from graspy.plot import heatmap
from src.cluster import get_paired_inds  # TODO fix the location of this func
from src.data import load_metagraph
from src.graph import preprocess
from src.io import savecsv, savefig
from src.utils import invert_permutation
from src.visualization import CLASS_COLOR_DICT, adjplot

print(scipy.__version__)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


graph_type = "G"
master_mg = load_metagraph(graph_type, version="2020-04-23")
mg = preprocess(
    master_mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

degrees = mg.calculate_degrees()
quant_val = np.quantile(degrees["Total edgesum"], 0.05)

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > quant_val].index
print(quant_val)
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

idx = mg.meta[mg.meta["Pair"].isin(mg.meta.index)].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)

meta = mg.meta
meta["pair_td"] = meta["Pair ID"].map(meta.groupby("Pair ID")["Total degree"].mean())
mg = mg.sort_values(["pair_td", "Pair ID"], ascending=False)
meta["inds"] = range(len(meta))
adj = mg.adj.copy()
lp_inds, rp_inds = get_paired_inds(meta)
left_inds = meta[meta["left"]]["inds"]


# %%


from src.hierarchy import signal_flow

sf = signal_flow(adj)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(x=sf[lp_inds], y=sf[rp_inds], ax=ax, s=15, linewidth=0, alpha=0.8)
corr = np.corrcoef(sf[lp_inds], sf[rp_inds])[0, 1]
ax.text(0.75, 0.05, f"Corr. = {corr:.2f}", transform=ax.transAxes)
ax.set_xlabel("Left")
ax.set_ylabel("Right")
ax.set_title("Pair signal flow")
stashfig("pair-signal-flow")
