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


shuffle_inds = np.random.permutation(n_verts)
B = A[np.ix_(shuffle_inds, shuffle_inds)]

faq = FastApproximateQAP(
    max_iter=30,
    eps=0.0001,
    init_method="rand",
    n_init=10,
    shuffle_input=False,
    maximize=True,
)

A_found, B_found = faq.fit_predict(A, B)
perm_inds = faq.perm_inds_

heatmap(
    A - B_found, title="Diff between true and FAQ-prediced adjacency", vmin=-1, vmax=1
)


# %%
from sgm import ScipyJVClassicSGM, JVSparseSGM

from graspy.match import SinkhornKnopp


def doubly_stochastic(n, barycenter=False):
    sk = SinkhornKnopp()
    K = np.random.rand(
        n, n
    )  # generate a nxn matrix where each entry is a random integer [0,1]
    for i in range(10):  # perform 10 iterations of Sinkhorn balancing
        K = sk.fit(K)
    if barycenter:
        J = np.ones((n, n)) / float(n)  # initialize J, a doubly stochastic barycenter
        P = (K + J) / 2
    else:
        P = K
    return P


doubly_stochastic(10)

#%%

from scipy.sparse import csr_matrix

n_verts = A.shape[0]
n_sims = 10

A = csr_matrix(A)
B = csr_matrix(A)
# P = csr_matrix(P)


for i in range(n_sims):
    P = doubly_stochastic(n_verts, barycenter=False)
    P = csr_matrix(P)
    sgm = JVSparseSGM(A, B, P)
    node_map = sgm.run(num_iters=100, tolerance=0, verbose=True)
    P_out = csr_matrix((np.ones(n_verts), (np.arange(n_verts), node_map)))
    B_out = P_out @ B @ P_out.T
    print((A != B_out).sum())
    B_out = B_out.todense()
    # heatmap(A.todense() - B_out)

A = A.todense()
# B_out = B_out.todense()
heatmap(B_out)
heatmap(A)


# %%
#!/usr/bin/env python

"""
    examples/synthetic/main.py
"""

import sys
import numpy as np
from scipy import sparse

from sgm import JVSparseSGM
from 

def make_perm(num_nodes, num_seeds):
    P = sparse.eye(num_nodes).tocsr()

    perm = np.arange(num_nodes)
    perm[num_seeds:] = np.random.permutation(perm[num_seeds:])

    return P[perm]


def make_init(num_nodes, num_seeds):
    P = sparse.csr_matrix((num_nodes, num_nodes))
    # P[:num_seeds, :num_seeds] = sparse.eye(num_seeds)
    return P


# --
# Create data

num_nodes = 128
num_seeds = 0

# Random symmetric matrix
A = sparse.random(num_nodes, num_nodes, density=0.1)
A = ((A + A.T) > 0).astype(np.float32)

# Random permutation matrix that keeps first `num_seeds` nodes the same
P_act = make_perm(num_nodes=num_nodes, num_seeds=num_seeds)

# Permute A according to P_act
B = P_act @ A @ P_act.T

assert (A[:num_nodes, :num_nodes] != B[:num_nodes, :num_nodes]).sum() > 0
assert (A[:num_seeds, :num_seeds] != B[:num_seeds, :num_seeds]).sum() == 0

# --
# Run SGM

P_init = make_init(num_nodes=num_nodes, num_seeds=num_seeds)

n_sims = 100
best_num_disagreements = np.inf
best_B = 0
for i in range(n_sims):
    P_init = doubly_stochastic(num_nodes)
    P_init = csr_matrix(P_init)
    sgm = JVSparseSGM(A=A, B=B, P=P_init, verbose=False)
    node_map = sgm.run(num_iters=100, tolerance=10)
    P_out = sparse.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), node_map)))
    B_perm = P_out @ B @ P_out.T
    num_disagreements = (
        A[:num_nodes, :num_nodes] != B_perm[:num_nodes, :num_nodes]
    ).sum()
    print("num_disagreements=%d" % num_disagreements)
    n_edges = A.sum()
    print(f"Proportional: {num_disagreements / (2*n_edges)}")
    if num_disagreements < best_num_disagreements:
        best_B = B_perm

    if num_disagreements == 0:
        break

heatmap(A.todense() - best_B.todense(), vmin=-1, vmax=1)

#%%
# --
# Check number of disagreements after SGM


heatmap(A.todense()[:100, :100])
heatmap(B_perm.todense()[:100, :100])


# If worked perfectly, `P_out @ P_act` should be identity matrix
# ((P_out @ P_act) != sparse.eye(num_nodes)).sum()


# %%
from sgm import ScipyJVClassicSGM
sgm = ScipyJVClassicSGM(A, B, P_init)
sgm.run(num_iters=100, tolerance=10)

# %%
