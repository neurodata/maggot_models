# %% [markdown]
# ## Imports and functions

import os
from operator import itemgetter
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy import version

from graspy.embed import LaplacianSpectralEmbed
from graspy.match import FastApproximateQAP
from graspy.plot import heatmap, pairplot
from graspy.simulations import er_np, sbm
from graspy.utils import get_lcc
from src.data import load_everything
from src.match import GraphMatch
from src.utils import savefig

print(version)
print(scipy.__version__)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = True
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


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


def n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def get_template_mat(A):
    total_synapses = np.sum(A)
    upper_triu_inds = np.triu_indices_from(A, k=1)
    filler = total_synapses / len(upper_triu_inds[0])
    upper_triu_template = np.zeros_like(A)
    upper_triu_template[upper_triu_inds] = filler
    return upper_triu_template


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


# %% [markdown]
# ## Generate a "perfect" feedforward network (stochastic block model)
low_p = 0.01
diag_p = 0.1
feedforward_p = 0.2
community_sizes = np.array(5 * [20])
block_probs = get_feedforward_B(low_p, diag_p, feedforward_p)
A = sbm(community_sizes, block_probs, directed=True, loops=False)
n_verts = A.shape[0]


plt.figure(figsize=(10, 10))
plt.title("Feedforward SBM block probability matrix")
sns.heatmap(block_probs, annot=True, square=True, cmap="Reds", cbar=False)
stashfig("ffwSBM-B")
plt.show()

heatmap(A, cbar=False, title="Feedforward SBM sampled adjacency matrix")
stashfig("ffwSBM-adj")
plt.show()

labels = n_to_labels(community_sizes).astype(str)

# %% [markdown]
# # Demonstrate that FAQ works
# Shuffle the true adjacency matrix and then show that it can be recovered
n_verts = 100
p = 0.1
n_init = 10
n_iter = 30
tol = 100
eps = 0.0001

A = er_np(n_verts, p=p)
shuffle_inds = np.random.permutation(n_verts)
B = A[np.ix_(shuffle_inds, shuffle_inds)]

faq = FastApproximateQAP(
    max_iter=n_iter,
    eps=eps,
    init_method="rand",
    n_init=n_init,
    shuffle_input=False,
    maximize=True,
)

start = timer()
A_found, B_found = faq.fit_predict(A, B)
diff = (timer() - start) / 60
print(f"Ali took {diff} minutes")
faq_score = np.abs(A - B_found).sum()
print(faq_score)


gm = GraphMatch(n_init=10 * n_init, n_iter=n_iter, tol=1000, solver="sparse")
start = timer()
B_found, perm_inds = gm.fit_predict(A, B)
diff = (timer() - start) / 60
print(f"BKJ-normal took {diff} minutes")

gm_score = np.abs(A - B_found).sum()
print(gm_score)

