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

from scipy import version
import scipy

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

# A[:20, 20:40] *= 2
# A[20:40, 40:60] *= 3
# A[40:60, 60:80] *= 4
# A[60:80, 80:100] *= 5


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
stashfig("faq-works")


#%%
inv_shuffle_inds = invert_permutation(shuffle_inds)
perm_inds

arr = np.empty((n_verts))
arr[inv_shuffle_inds] = perm_inds
arr

scatter_df = pd.DataFrame()
scatter_df["True Ind"] = range(n_verts)
scatter_df["Pred Ind"] = arr
scatter_df["Score"] = faq.score_

plt.figure(figsize=(10, 10))
sns.scatterplot(data=scatter_df, x="True Ind", y="Pred Ind", hue="Score")
plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.0)
# %% [markdown]
# # Use multiple restarts of FAQ and a template upper triangular matrix
template = get_template_mat(A)
n_init = 1
faq = FastApproximateQAP(
    max_iter=20,
    eps=0.0001,
    init_method="rand",
    n_init=100,
    shuffle_input=False,
    maximize=True,
)

fig, axs = plt.subplots(5, 4, figsize=(20, 20))
axs = axs.ravel()

perm_inds_mat = np.zeros((n_init, n_verts))
scores = []
dfs = []
for i in range(n_init):
    print()
    print(i)
    print()

    # shuffle A
    shuffle_inds = np.random.permutation(n_verts)
    A_shuffle = A[np.ix_(shuffle_inds, shuffle_inds)].copy()

    # fit FAQ
    _, A_found = faq.fit_predict(template, A_shuffle)
    temp_perm_inds = faq.perm_inds_
    heatmap(A_shuffle[np.ix_(temp_perm_inds, temp_perm_inds)], cbar=False, ax=axs[i])

    # put things back in order
    pred_inds = np.empty_like(shuffle_inds)
    pred_inds[shuffle_inds[temp_perm_inds]] = range(n_verts)
    perm_inds_mat[i, :] = pred_inds

    temp_df = pd.DataFrame()
    temp_df["True Ind"] = range(n_verts)
    temp_df["Predicted Ind"] = pred_inds
    temp_df["Score"] = faq.score_
    temp_df["Labels"] = labels
    dfs.append(temp_df)
    scores.append(faq.score_)

plt.tight_layout()
stashfig("multisort-heatmap")
plt.show()

#%%
scatter_df = pd.concat(dfs)

plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=scatter_df, x="True Ind", y="Predicted Ind", hue="Score", palette="Blues_r"
)
median_df = pd.DataFrame()
median_df["True Ind"] = range(n_verts)
median_df["Median Predicted Ind"] = np.median(perm_inds_mat, axis=0)
median_df["Labels"] = labels
sns.scatterplot(
    data=median_df, x="True Ind", y="Median Predicted Ind", marker="s", color="r"
)
plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.0)
stashfig("true-vs-pred-UT-match")


# %% [markdown]
# # add the varshney stuff for comparison
signal_df = signal_flow(A)
labels = labels.astype(str)
signal_df["Labels"] = labels
signal_df["Metric name"] = "Signal flow"
sf = signal_df["Signal flow"]
sf -= sf.min()
sf /= sf.max()
print(sf.max())
signal_df["Metric"] = sf
median_df["Metric name"] = "UT Match"
median_df["Metric"] = 1 - median_df["Median Predicted Ind"] / n_verts

dist_df = pd.concat((signal_df, median_df))

fg = sns.FacetGrid(
    dist_df,
    row="Labels",
    col="Metric name",
    aspect=2,
    hue="Labels",
    hue_order=["0", "1", "2", "3", "4"],
    palette=sns.color_palette("Set1", dist_df.Labels.nunique()),
    sharex=True,
    sharey=False,
    margin_titles=True,
)
fg.map(sns.distplot, "Metric", bins=np.linspace(0, 1, 30), kde=True)
fg.set(yticks=[], yticklabels=[])
stashfig("Ut-match-vs-signal-flow")


#%% junk beware
# # %%
# fg = sns.FacetGrid(
#     scatter_df,
#     row="Labels",
#     aspect=2,
#     hue="Labels",
#     hue_order=["0", "1", "2", "3", "4"],
#     palette=sns.color_palette("Set1", scatter_df.Labels.nunique()),
# )
# fg.map(sns.distplot, "Signal flow")
# fg.set(yticks=[], yticklabels=[])

# #%%
# # plt.figure(figsize=(10, 10))
# # sns.scatterplot(
# #     x="Lap-2",
# #     y="Signal flow",
# #     data=scatter_df,
# #     hue="Labels",
# #     palette=sns.color_palette("Set1", scatter_df.Labels.nunique()),
# # )
# # stashfig("lap2-sf-ffwSBM")
# # plt.show()
# scatter_df
# sns.set_context("talk", font_scale=1.5)
# fg = sns.FacetGrid(
#     scatter_df,
#     row="Labels",
#     aspect=2,
#     hue="Labels",
#     hue_order=["0", "1", "2", "3", "4"],
#     palette=sns.color_palette("Set1", scatter_df.Labels.nunique()),
# )
# fg.map(sns.distplot, "Signal flow")
# fg.set(yticks=[], yticklabels=[])


# # triang_df = pd.DataFrame()
# # triang_df["Mean perm ind"] = mean_perm_ind
# # triang_df["Labels"] = labels[shuffle_inds]

# fg = sns.FacetGrid(
#     median_df,
#     row="Labels",
#     aspect=2,
#     hue="Labels",
#     hue_order=["0", "1", "2", "3", "4"],
#     palette=sns.color_palette("Set1", scatter_df.Labels.nunique()),
# )
# fg.map(sns.distplot, "Median Predicted Ind")
# fg.set(yticks=[], yticklabels=[])


# # %% [markdown]
# # ## Now, look at the output for this signal flow metric on the A $\rightarrow$ D graph
# # Here I am just using labels for MB and PNs, as well as indicating side with the
# # direction of the marker

# GRAPH_VERSION = "2019-09-18-v2"
# adj, class_labels, side_labels = load_everything(
#     "Gad", GRAPH_VERSION, return_class=True, return_side=True
# )

# adj, inds = get_lcc(adj, return_inds=True)
# class_labels = class_labels[inds]
# side_labels = side_labels[inds]

# name_map = {" mw right": "R", " mw left": "L"}
# side_labels = np.array(itemgetter(*side_labels)(name_map))

# name_map = {
#     "CN": "Unk",
#     "DANs": "MBIN",
#     "KCs": "KC",
#     "LHN": "Unk",
#     "LHN; CN": "Unk",
#     "MBINs": "MBIN",
#     "MBON": "MBON",
#     "MBON; CN": "MBON",
#     "OANs": "MBIN",
#     "ORN mPNs": "mPN",
#     "ORN uPNs": "uPN",
#     "tPNs": "tPN",
#     "vPNs": "vPN",
#     "Unidentified": "Unk",
#     "Other": "Unk",
# }
# class_labels = np.array(itemgetter(*class_labels)(name_map))


# total_synapses = np.sum(adj)
# upper_triu_inds = np.triu_indices_from(adj, k=1)
# filler = total_synapses / len(upper_triu_inds[0])
# upper_triu_template = np.zeros_like(adj)
# upper_triu_template[upper_triu_inds] = filler

# shuffle_inds = np.random.permutation(adj.shape[0])
# B = adj[np.ix_(shuffle_inds, shuffle_inds)]

# faq = FastApproximateQAP(shuffle_input=False, max_iter=1, init_method="rand")
# pred_sort_inds = faq.fit_predict(upper_triu_template, B)

# P = np.zeros(B.shape)
# P[range(B.shape[0]), pred_sort_inds] = 1
# heatmap(upper_triu_template)
# heatmap(P @ B @ P.T)

# # %% [markdown]
# # # JUNK

# # %% [markdown]
# # #

# perm_inds = faq.perm_inds_
# # perm_inds

# # indicator = np.zeros((n_verts, n_verts))
# # indicator += np.diag(np.arange(n_verts))
# # indicator

# # order = np.diag(P_found @ indicator @ P_found.T).astype(int)
# # print(order)

# # heatmap(A - B[np.ix_(order, order)])
# # heatmap(A - B[np.ix_(perm_inds, perm_inds)])
# # heatmap(A - B)

# #%%

# heatmap(A, cbar=False, title="Feedforward SBM adjacency matrix")

# heatmap(B, cbar=False, title="Shuffled SBM adjacency matrix")

# heatmap(upper_triu_template, title="Template upper triangular matrix")

# heatmap(B[np.ix_(perm_inds, perm_inds)], title="FAQ predicted sortings", cbar=False)
# heatmap(B_found, title="FAQ predicted sortings", cbar=False)
# heatmap(A - B_found, title="Diff", cbar=False)
