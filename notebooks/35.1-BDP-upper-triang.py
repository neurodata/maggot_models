# %% [markdown]
# ## Imports and functions

import os
from operator import itemgetter
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy import version

from graspy.match import FastApproximateQAP
from graspy.plot import heatmap
from graspy.simulations import sbm
from graspy.utils import get_lcc
from src.data import load_everything
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


def signal_flow(A, n_components=5, return_evals=False):
    """ Implementation of the signal flow metric from Varshney et al 2011
    
    Parameters
    ----------
    A : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    D_root = np.diag(np.diag(D) ** (-1 / 2))
    D_root[np.isnan(D_root)] = 0
    D_root[np.isinf(D_root)] = 0
    Q = D_root @ L @ D_root
    evals, evecs = np.linalg.eig(Q)
    inds = np.argsort(evals)
    evals = evals[inds]
    evecs = evecs[:, inds]
    evecs = np.diag(np.diag(D) ** (1 / 2)) @ evecs
    # return evals, evecs, z, D_root
    scatter_df = pd.DataFrame()
    for i in range(1, n_components + 1):
        scatter_df[f"Lap-{i+1}"] = evecs[:, i]
    scatter_df["Signal flow"] = z
    if return_evals:
        return scatter_df, evals
    else:
        return scatter_df


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
# # Try on real data

print("Trying on real data")


def unshuffle(shuffle_inds, perm_inds):
    pred_inds = np.empty_like(shuffle_inds)
    pred_inds[shuffle_inds[perm_inds]] = range(len(shuffle_inds))
    return pred_inds


GRAPH_VERSION = "2019-09-18-v2"
adj, class_labels, side_labels = load_everything(
    "Gad", GRAPH_VERSION, return_keys=["Class", "Hemisphere"]
)

adj, inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[inds]
side_labels = side_labels[inds]

n_verts = adj.shape[0]

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

template = get_template_mat(adj)

# shuffle_inds = np.random.permutation(n_verts)

# shuffle_adj = adj[np.ix_(shuffle_inds, shuffle_inds)]

n_init = 1
faq = FastApproximateQAP(
    max_iter=30,
    eps=0.0001,
    init_method="rand",
    n_init=n_init,
    shuffle_input=True,
    gmp=True,
)
start = timer()
faq.fit(template, adj)
end = timer()
print(f"FAQ took {(end - start)/60.0} minutes")

perm_inds = faq.perm_inds_
# perm_inds = unshuffle(shuffle_inds, shuffle_perm_inds)
# %% [markdown]
# #
from graspy.plot import gridplot

gridplot([adj[np.ix_(perm_inds, perm_inds)]])
stashfig("unshuffled-real-heatmap-faq")

# %% [markdown]
# #
from src.hierarchy import signal_flow

z = signal_flow(adj)
sort_inds = np.argsort(z)[::-1]
gridplot([adj[np.ix_(sort_inds, sort_inds)]])
stashfig("unshuffled-real-heatmap-sf")


# %% [markdown]
# #

# %% [markdown]
# #


def shuffle_edges(A):
    n_verts = A.shape[0]
    A_fake = A.copy().ravel()
    np.random.shuffle(A_fake)
    A_fake = A_fake.reshape((n_verts, n_verts))
    return A_fake


fake_adj = shuffle_edges(adj)
n_init = 1
faq = FastApproximateQAP(
    max_iter=30,
    eps=0.0001,
    init_method="rand",
    n_init=n_init,
    shuffle_input=True,
    gmp=True,
)
start = timer()
faq.fit(template, fake_adj)
end = timer()
print(f"FAQ took {(end - start)/60.0} minutes")


fake_perm_inds = faq.perm_inds_
gridplot([fake_adj[np.ix_(perm_inds, perm_inds)]])
stashfig("fake-heatmap-faq")

