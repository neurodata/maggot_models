# %% [markdown]
# # Load and import
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import binarize, get_lcc
from src.data import load_everything
from src.hierarchy import normalized_laplacian, signal_flow
from src.utils import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = False
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


GRAPH_VERSION = "2019-09-18-v2"
adj, class_labels, side_labels = load_everything(
    "Gad", GRAPH_VERSION, return_class=True, return_side=True
)

adj, inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[inds]
side_labels = side_labels[inds]
n_verts = adj.shape[0]

# %% [markdown]
# # Sort shuffled AD graph on signal flow
fake_adj = adj.copy()
np.random.shuffle(fake_adj.ravel())
fake_adj = fake_adj.reshape((n_verts, n_verts))

z = signal_flow(fake_adj)
sort_inds = np.argsort(z)[::-1]

gridplot([fake_adj[np.ix_(sort_inds, sort_inds)]], height=20)
stashfig("gridplot-sf-sorted-fake")


# %% [markdown]
# # Sort true AD graph on signal flow

z = signal_flow(adj)
sort_inds = np.argsort(z)[::-1]

gridplot([adj[np.ix_(sort_inds, sort_inds)]], height=20)
stashfig("gridplot-sf-sorted")


# %% [markdown]
# # Look at graph laplacians

evecs, evals = normalized_laplacian(
    adj, n_components=5, return_evals=True, normalize_evecs=True
)


scatter_mat = np.concatenate((z[:, np.newaxis], evecs), axis=1)
pairplot(scatter_mat, labels=class_labels, palette="tab20")

# %% [markdown]
# # Examine the 2nd eigenvector
degree = ((adj + adj.T) / 2).sum(axis=1)

evecs, evals = normalized_laplacian(
    adj, n_components=2, return_evals=True, normalize_evecs=True
)
evec2_norm = evecs[:, 1]

evecs, evals = normalized_laplacian(
    adj, n_components=2, return_evals=True, normalize_evecs=False
)
evec2_unnorm = evecs[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(evec2_norm)

plt.figure(figsize=(10, 6))
plt.plot(evec2_unnorm)

plt.figure(figsize=(10, 6))
plt.scatter(evec2_norm, degree)

plt.figure(figsize=(10, 6))
plt.scatter(evec2_unnorm, degree)

