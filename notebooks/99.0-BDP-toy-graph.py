#%%
from graspy.simulations import sbm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n_per_comm = np.array([20, 20, 40, 30, 40, 20, 50, 30, 20, 30, 20])
n_comm = len(n_per_comm)

base_p = 0.005
P = np.full((n_comm, n_comm), base_p)

P[0, 3] = 0.2
P[1, 4] = 0.25
P[2, 4] = 0.2
P[2, 5] = 0.15
P[3, 4] = 0.1
P[4, 3] = 0.06
P[3, 6] = 0.25
P[3, 9] = 0.05
P[4, 7] = 0.2
P[4, 8] = 0.35
P[7, 8] = 0.05
P[7, 4] = 0.02
P[7, 10] = 0.2
P[8, 10] = 0.15
P[7, 9] = 0.05
P[6, 9] = 0.2
P[9, 6] = 0.05

weighted = True
if weighted:
    wt = n_comm * [n_comm * [np.random.poisson]]
    lams = np.random.uniform(0.1, 0.3, size=(n_comm ** 2))
    lams = lams.reshape(n_comm, n_comm)
    lams[P != base_p] = P[P != base_p] * 5
    wtargs = np.array([dict(lam=lam) for lam in lams.ravel()]).reshape(n_comm, n_comm)
else:
    lams = np.ones_like(P)
    wtargs = None
    wt = 1

adj, labels = sbm(
    n_per_comm, P, directed=True, loops=False, wt=wt, wtargs=wtargs, return_labels=True
)


sns.set_context("talk", font_scale=0.5)
fig, axs = plt.subplots(1, 3, figsize=(16, 8))
sns.heatmap(
    P,
    annot=True,
    square=True,
    ax=axs[0],
    cbar=False,
    # cbar_kws=dict(shrink=0.7),
    cmap="RdBu_r",
    center=0,
)
sns.heatmap(
    lams.reshape(n_comm, n_comm),
    square=True,
    ax=axs[1],
    cbar=False,
    cmap="RdBu_r",
    center=0,
)


sns.heatmap(adj, ax=axs[2], square=True, cbar=False, cmap="RdBu_r", center=0)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

from graspy.plot import heatmap

heatmap(adj, inner_hier_labels=labels, transform="simple-all")

# %% [markdown]
# #

import networkx as nx

minigraph = nx.from_numpy_array(P, nodelist=labels)

# %% [markdown]
# #


from src.traverse import to_markov_matrix

prob_mat = to_markov_matrix(adj)

prob_powers = [np.linalg.matrix_power(prob_mat, n) for n in range(1, 11)]


n_timesteps = 10
n_verts = np.sum(n_per_comm)
uni_labels = np.unique(labels)
prob_powers = np.array(prob_powers)
hist_mat = np.empty((n_verts, n_timesteps * len(uni_labels)))
for i, ul in enumerate(uni_labels):
    from_inds = np.where(labels == ul)[0]
    activations = prob_powers[:, from_inds, :]
    total_activation = activations.sum(axis=1)
    hist_mat[:, i * n_timesteps : (i + 1) * n_timesteps] = total_activation.T

plot_hist_mat = np.log10(hist_mat + 1)
# plot_hist_mat = hist_mat
color_dict = dict(zip(uni_labels, sns.color_palette("tab10")))
colors = np.array(np.vectorize(color_dict.get)(labels)).T
cg = sns.clustermap(
    data=plot_hist_mat,
    col_cluster=False,
    row_colors=colors,
    cmap="RdBu_r",
    center=0,
    cbar_pos=None,
    method="average",
    metric="euclidean",
)

ax = cg.ax_heatmap
bins = np.arange(1, n_timesteps + 1)
xs = np.arange(bins.shape[0] - 0.5, hist_mat.shape[-1] - 1, step=bins.shape[0])
ticks = []
for i, x in enumerate(xs):
    ax.axvline(x, linestyle="--", linewidth=1, color="grey")
    ticks.append(x - 0.5 * n_timesteps)
ticks.append(ticks[-1] + n_timesteps)
ax.set_xticks(ticks)
ax.set_xticklabels(uni_labels)

ax.set_xlabel("Response over time")
ax.set_ylabel("Neuron")
# ax.set_title(f"metric={metric}, linkage={linkage}")
