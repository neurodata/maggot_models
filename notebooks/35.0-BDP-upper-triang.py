# %% [markdown]
# # Signal flow of the Drosophila larva brain

# %% [markdown]
# ## Imports and functions

import os
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.embed import LaplacianSpectralEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import sbm
from graspy.utils import get_lcc
from graspy.match import FastApproximateQAP

from src.data import load_everything
from src.utils import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = False
DEFAULT_FMT = "png"
DEFUALT_DPI = 150


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)

# %% [markdown]
# ## Generate a "perfect" feedforward network (stochastic block model)
low_p = 0
diag_p = 0
feedforward_p = 0.2
community_sizes = 5 * [20]
B = get_feedforward_B(low_p, diag_p, feedforward_p)

plt.figure(figsize=(10, 10))
plt.title("Feedforward SBM block probability matrix")
sns.heatmap(B, annot=True, square=True, cmap="Reds", cbar=False)
stashfig("ffwSBM-B")
plt.show()

A = sbm(community_sizes, B, directed=True, loops=False)
heatmap(A, cbar=False, title="Feedforward SBM sampled adjacency matrix")
stashfig("ffwSBM-adj")
plt.show()

# %% [markdown]
# # try upper triangular thing
total_synapses = np.sum(A)
upper_triu_inds = np.triu_indices_from(A, k=1)
filler = total_synapses / len(upper_triu_inds[0])
upper_triu_template = np.zeros_like(A)
upper_triu_template[upper_triu_inds] = filler

shuffle_inds = np.random.permutation(A.shape[0])

faq = FastApproximateQAP(shuffle_input=False, n_init=20, init_method="rand")
B = A[np.ix_(shuffle_inds, shuffle_inds)]
pred_sort_inds = faq.fit_predict(upper_triu_template, B)

heatmap(upper_triu_template)
P = np.zeros(A.shape)
P[range(A.shape[0]), pred_sort_inds] = 1

heatmap(P @ B @ P.T)
# %% [markdown]
# ## Now, look at the output for this signal flow metric on the A $\rightarrow$ D graph
# Here I am just using labels for MB and PNs, as well as indicating side with the
# direction of the marker

GRAPH_VERSION = "2019-09-18-v2"
adj, class_labels, side_labels = load_everything(
    "Gad", GRAPH_VERSION, return_class=True, return_side=True
)

adj, inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[inds]
side_labels = side_labels[inds]

name_map = {" mw right": "R", " mw left": "L"}
side_labels = np.array(itemgetter(*side_labels)(name_map))

name_map = {
    "CN": "Unk",
    "DANs": "MBIN",
    "KCs": "KC",
    "LHN": "Unk",
    "LHN; CN": "Unk",
    "MBINs": "MBIN",
    "MBON": "MBON",
    "MBON; CN": "MBON",
    "OANs": "MBIN",
    "ORN mPNs": "mPN",
    "ORN uPNs": "uPN",
    "tPNs": "tPN",
    "vPNs": "vPN",
    "Unidentified": "Unk",
    "Other": "Unk",
}
class_labels = np.array(itemgetter(*class_labels)(name_map))


total_synapses = np.sum(adj)
upper_triu_inds = np.triu_indices_from(adj, k=1)
filler = total_synapses / len(upper_triu_inds[0])
upper_triu_template = np.zeros_like(adj)
upper_triu_template[upper_triu_inds] = filler

shuffle_inds = np.random.permutation(adj.shape[0])
B = adj[np.ix_(shuffle_inds, shuffle_inds)]

faq = FastApproximateQAP(shuffle_input=False, max_iter=1, init_method="rand")
pred_sort_inds = faq.fit_predict(upper_triu_template, B)

P = np.zeros(B.shape)
P[range(B.shape[0]), pred_sort_inds] = 1
heatmap(upper_triu_template)
heatmap(P @ B @ P.T)

