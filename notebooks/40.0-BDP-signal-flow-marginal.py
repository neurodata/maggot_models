# %% [markdown]
# # try graph flow
import json
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks, remove_loops
from src.data import load_everything
from src.hierarchy import signal_flow
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

BRAIN_VERSION = "2019-12-09"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

SAVEFIGS = False
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

SAVESKELS = False
SAVEOBJS = False


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


def signal_flow_marginal(adj, labels, col_wrap=5, palette="tab20"):
    sf = signal_flow(adj)
    uni_labels = np.unique(labels)
    medians = []
    for i in uni_labels:
        inds = np.where(labels == i)[0]
        medians.append(np.median(sf[inds]))
    sort_inds = np.argsort(medians)[::-1]
    col_order = uni_labels[sort_inds]
    plot_df = pd.DataFrame()
    plot_df["Signal flow"] = sf
    plot_df["Class"] = labels
    fg = sns.FacetGrid(
        plot_df,
        col="Class",
        aspect=1.5,
        palette=palette,
        col_order=col_order,
        sharey=False,
        col_wrap=col_wrap,
        xlim=(-3, 3),
    )
    fg = fg.map(sns.distplot, "Signal flow")  # bins=np.linspace(-2.2, 2.2))
    fg.set(yticks=[], yticklabels=[])
    plt.tight_layout()
    return fg


def weighted_signal_flow(A):
    """Implementation of the signal flow metric from Varshney et al 2011
    
    Parameters
    ----------
    A : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * (A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


# %% [markdown]
# # Load data

adj, class_labels, side_labels, skeleton_labels = load_everything(
    "Gad",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere"],
    return_ids=True,
)

sf = signal_flow(adj)

# %% [markdown]
# # Compute signal flow marginals for known cell types

signal_flow_marginal(adj, class_labels)
stashfig("known-class-sf-marginal")


# %% [markdown]
# # Write out signal flow as color for jsons


norm = colors.Normalize(vmin=sf.min(), vmax=sf.max())
sm = ScalarMappable(norm=norm, cmap="Reds")
cmap = sm.to_hex(sf)

export_skeleton_json("signal-flow", skeleton_labels, cmap, palette=None)

# # %% [markdown]
# # #
# node_signal_flow = signal_flow(adj)
# mean_sf = np.zeros(k)
# for i in np.unique(pred_labels):
#     inds = np.where(pred_labels == i)[0]
#     mean_sf[i] = np.mean(node_signal_flow[inds])

# cluster_mean_latent = gmm.model_.means_[:, 0]
# block_probs = SBMEstimator().fit(bin_adj, y=pred_labels).block_p_
# block_prob_df = pd.DataFrame(data=block_probs, index=range(k), columns=range(k))
# block_g = nx.from_pandas_adjacency(block_prob_df, create_using=nx.DiGraph)
# plt.figure(figsize=(10, 10))
# # don't ever let em tell you you're too pythonic
# pos = dict(zip(range(k), zip(cluster_mean_latent, mean_sf)))
# # nx.draw_networkx_nodes(block_g, pos=pos)
# labels = nx.get_edge_attributes(block_g, "weight")
# # nx.draw_networkx_edge_labels(block_g, pos, edge_labels=labels)
# from matplotlib.cm import ScalarMappable
# import matplotlib as mpl

# norm = mpl.colors.LogNorm(vmin=0.01, vmax=0.1)

# sm = ScalarMappable(cmap="Reds", norm=norm)
# cmap = sm.to_rgba(np.array(list(labels.values())) + 0.01)
# nx.draw_networkx(
#     block_g,
#     pos,
#     edge_cmap="Reds",
#     edge_color=cmap,
#     connectionstyle="arc3,rad=0.2",
#     width=1.5,
# )


# %%
