#%%
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap
from src.data import load_right

# Load data
right_adj, right_labels = load_right()


# Fit the models
sbm = SBMEstimator(directed=True, loops=False)
sbm.fit(right_adj, y=right_labels)

dcsbm = DCSBMEstimator(degree_directed=False, directed=True, loops=False)
dcsbm.fit(right_adj, y=right_labels)

rdpg = RDPGEstimator(loops=False, n_components=3)
rdpg.fit(right_adj)

# Plotting
np.random.seed(8888)

cmap = mpl.cm.get_cmap("RdBu_r")
center = 0
vmin = 0
vmax = 1
norm = mpl.colors.Normalize(0, 1)
cc = np.linspace(0.5, 1, 256)
cmap = mpl.colors.ListedColormap(cmap(cc))

heatmap_kws = dict(
    cbar=False,
    font_scale=1.4,
    vmin=0,
    vmax=1,
    inner_hier_labels=right_labels,
    hier_label_fontsize=16,
    cmap=cmap,
    center=None,
)
side_label_kws = dict(labelpad=45, fontsize=24)


fig, ax = plt.subplots(3, 2, figsize=(10, 17))

# SBM
heatmap(sbm.p_mat_, ax=ax[0, 0], title="Probability matrix", **heatmap_kws)
heatmap(np.squeeze(sbm.sample()), ax=ax[0, 1], title="Random sample", **heatmap_kws)
ax[0, 0].set_ylabel("SBM", **side_label_kws)

# DCSBM
heatmap(dcsbm.p_mat_, ax=ax[1, 0], **heatmap_kws)
heatmap(np.squeeze(dcsbm.sample()), ax=ax[1, 1], **heatmap_kws)
ax[1, 0].set_ylabel("DCSBM", **side_label_kws)

# RDPG
heatmap(rdpg.p_mat_, ax=ax[2, 0], **heatmap_kws)
heatmap(np.squeeze(rdpg.sample()), ax=ax[2, 1], **heatmap_kws)
ax[2, 0].set_ylabel("RDPG", **side_label_kws)

plt.tight_layout()

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(rdpg.p_mat_)
cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", pad=0.01, shrink=0.8, fraction=0.1
)
cbar.ax.tick_params(labelsize=16)

plt.savefig(
    "./maggot_models/reports/figures/explain_models/explain_models.pdf",
    facecolor="w",
    format="pdf",
    bbox_inches="tight",
)

