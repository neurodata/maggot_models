# %% [markdown]
# #
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.simulations import sbm
from src.io import savefig
from graspy.plot import heatmap


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


low_p = 0.01
diag_p = 0.2
feedforward_p = 0.5
n_blocks = 5


block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
plt.figure(figsize=(10, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False)
plt.title("Feedforward block probability matrix")

#%%
community_sizes = np.empty(2 * n_blocks, dtype=int)
n_feedforward = 10
community_sizes = n_blocks * [n_feedforward]

np.random.seed(88)
A, labels = sbm(
    community_sizes, block_probs, directed=True, loops=False, return_labels=True
)
n_verts = A.shape[0]

heatmap(A, inner_hier_labels=labels, cbar=False)

# %% [markdown]
# ##

from src.visualization import matrixplot


def _get_tick_info(sort_meta, sort_class):
    """ Assumes meta is already sorted
    """
    if sort_meta is not None and sort_class is not None:
        # get locations
        sort_meta["sort_idx"] = range(len(sort_meta))
        # for the gridlines
        print(sort_meta)
        print(sort_class)
        first_df = sort_meta.groupby(sort_class, sort=False).first()
        first_inds = list(first_df["sort_idx"].values)[1:]  # skip since we have spines
        # for the tic locs
        middle_df = sort_meta.groupby(sort_class, sort=False).mean()
        middle_inds = np.array(middle_df["sort_idx"].values) + 0.5
        middle_labels = list(middle_df.index)
        return first_inds, middle_inds, middle_labels
    else:
        return None, None, None


df = pd.DataFrame(data=labels)
df["inds"] = range(len(df))
df.groupby([0]).first()

_get_tick_info(df, [df.columns.values[0]])
#%%
sns.set_context("talk")
matrixplot(A, col_meta=labels, row_meta=labels)

# %% [markdown]
# ##
n_rows = 100
n_cols = 200
data11 = np.random.normal(0, 1, size=(n_rows, n_cols))
data21 = np.random.normal(0, 2, size=(n_rows, n_cols))
data12 = np.random.normal(1, 1, size=(n_rows, n_cols))
data22 = np.random.normal(1, 2, size=(n_rows, n_cols))
data = np.block([[data11, data12], [data21, data22]])

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(data, ax=ax)


# %% [markdown]
# ## Add row and column metadata
means = np.zeros(data.shape[0])
means[n_rows:] = 1

variances = np.ones(data.shape[1])
variances[n_cols:] = 2

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(data, ax=ax, row_meta=means, col_meta=variances)
ax.set_xlabel("Variance")
ax.set_ylabel("Mean")
