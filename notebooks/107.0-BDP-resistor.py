# %% [markdown]
# #

import os

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from graspy.plot import heatmap, pairplot

from graspy.utils import remove_loops
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels

from src.visualization import CLASS_COLOR_DICT, matrixplot

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


# %% [markdown]
# #
VERSION = "2020-04-01"
print(f"Using version {VERSION}")

graph_type = "Gad"
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
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

# %% [markdown]
# #

source_groups = [
    ("sens-ORN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-MN",),
    ("sens-thermo",),
    ("sens-vtd",),
    ("sens-AN",),
]
source_group_names = ["Odor", "Photo", "MN", "Temp", "VTD", "AN"]

sink_groups = [
    ("motor-mAN", "motormVAN", "motor-mPaN"),
    ("O_dSEZ", "O_dVNC;O_dSEZ", "O_dSEZ;CN", "LHN;O_dSEZ"),
    ("O_dVNC", "O_dVNC;CN", "O_RG;O_dVNC", "O_dVNC;O_dSEZ"),
    ("O_RG", "O_RG-IPC", "O_RG-ITP", "O_RG-CA-LP", "O_RG;O_dVNC"),
    ("O_dUnk",),
]
sink_group_names = ["Motor", "SEZ", "VNC", "RG", "dUnk"]

meta = mg.meta.copy()
class_key = "merge_class"

meta["idx"] = range(len(meta))


A = mg.adj
A = A.copy()
A = remove_loops(A)
W = (A + A.T) / 2
D = np.diag(np.sum(W, axis=1))
L = D - W
L_pinv = np.linalg.pinv(L)

voltage_df = pd.DataFrame(index=meta.index)
rank_voltage_df = pd.DataFrame(index=meta.index)
col_meta = pd.DataFrame(index=range(len(source_groups) * len(sink_groups)))
col_meta["in_out"] = -1

i = 0
for source_group, source_name in zip(source_groups, source_group_names):
    for sink_group, sink_name in zip(sink_groups, sink_group_names):
        from_inds = meta[meta[class_key].isin(source_group)]["idx"].values
        out_inds = meta[meta[class_key].isin(sink_group)]["idx"].values
        current = np.zeros((len(A), 1))
        current[from_inds] = 1 / len(from_inds)
        current[out_inds] = -1 / len(out_inds)
        v = L_pinv @ current
        # set the minimum voltage to 1
        v -= v.min()
        v += 1
        voltage_df[(source_name, sink_name)] = v
        rank_voltage_df[(source_name, sink_name)] = rankdata(v)
        col_meta.iloc[i, 0] = f"{source_name}" + r"$\to$ " f"{sink_name}"
        i += 1

        vdiff = np.squeeze(v[:, np.newaxis] - v[np.newaxis, :])
        curr = W * vdiff
        curr_node = np.sum(curr, axis=1)  # current from each node


fig, ax = plt.subplots(1, 1, figsize=(10, 20))
voltage = voltage_df.values
log_voltage = np.log10(voltage)
matrixplot(
    rank_voltage_df.values,
    ax=ax,
    row_meta=meta,
    row_sort_class=[class_key],
    col_meta=col_meta,
    col_sort_class=["in_out"],
    tick_rot=45,
)

# %% [markdown]
# #

pca = PCA(n_components=5)
embed = pca.fit_transform(rank_voltage_df.values)
pg = pairplot(embed, labels=meta[class_key].values, palette=CLASS_COLOR_DICT)
pg._legend.remove()

# %% [markdown]
# #
colors = np.vectorize(CLASS_COLOR_DICT.get)(meta["Merge Class"].values)
sns.clustermap(
    rank_voltage_df.values, row_cluster=True, col_cluster=False, row_colors=colors
)
