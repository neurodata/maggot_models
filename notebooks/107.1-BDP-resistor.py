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

from itertools import chain

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

sg = list(chain.from_iterable(source_groups))
og = list(chain.from_iterable(out_groups))
sg_name = "All"
og_name = "All"

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
# col_meta = pd.DataFrame(index=range(len(source_groups) * len(sink_groups)))
# col_meta["in_out"] = -1

from_inds = meta[meta[class_key].isin((sg[0],))]["idx"].values
out_inds = meta[meta[class_key].isin(og)]["idx"].values
current = np.zeros((len(A), 1))
current[from_inds] = 1 / len(from_inds)
current[out_inds] = -1 / len(out_inds)
v = L_pinv @ current
# set the minimum voltage to 1
v -= v.min()
v += 1
meta["voltage"] = -v
meta["rand"] = np.random.uniform(size=len(meta))


from src.visualization import adjplot

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.plot(
    (0, len(mg.adj)),
    (0, len(mg.adj)),
    linewidth=2,
    color="red",
    linestyle="--",
    alpha=0.5,
)
adjplot(
    mg.adj,
    meta=meta,
    item_order=["voltage", "rand"],
    plot_type="scattermap",
    ax=ax,
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    sizes=(1, 2),
)


# %% [markdown]
# ##
vdiff = np.squeeze(v[:, np.newaxis] - v[np.newaxis, :])
curr = np.abs(W * vdiff)
curr_node = np.sum(curr, axis=1)  # current from each node
meta["voltage"] = v
meta["curr"] = curr_node

sns.scatterplot(data=meta, x="curr", y="voltage")

# fig, ax = plt.subplots(1, 1, figsize=(10, 20))
# voltage = voltage_df.values
# log_voltage = np.log10(voltage)
# matrixplot(
#     rank_voltage_df.values,
#     ax=ax,
#     row_meta=meta,
#     row_sort_class=[class_key],
#     tick_rot=45,
# )

# %% [markdown]
# ##
sns.distplot(np.log10(curr_node + 1), kde=False)

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
