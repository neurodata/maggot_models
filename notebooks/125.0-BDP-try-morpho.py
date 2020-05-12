# %% [markdown]
# ##
from src.pymaid import start_instance
import pymaid
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    set_axes_equal,
    stacked_barplot,
)

mg = load_metagraph("G")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta


# %% [markdown]
# ##
start_instance()


# %% [markdown]
# #
import pandas as pd
import seaborn as sns

nl = pymaid.get_neurons(meta[meta["class1"] == "uPN"].index.values)
print(len(nl))

connectors = pymaid.get_connectors(nl)
connectors.set_index("connector_id", inplace=True)
connectors.drop(
    [
        "confidence",
        "creation_time",
        "edition_time",
        "tags",
        "creator",
        "editor",
        "type",
    ],
    inplace=True,
    axis=1,
)
details = pymaid.get_connector_details(connectors.index.values)
details.set_index("connector_id", inplace=True)
connectors = pd.concat((connectors, details), ignore_index=False, axis=1)
connectors.reset_index(inplace=True)

import matplotlib.pyplot as plt


from scipy.stats import gaussian_kde
import numpy as np


data = connectors
plot_vars = np.array(["x", "y", "z"])


data_mat = data[plot_vars].values

mins = []
maxs = []
for i in range(data_mat.shape[1]):
    dmin = data_mat[:, i].min()
    dmax = data_mat[:, i].max()
    mins.append(dmin)
    maxs.append(dmax)

print("making meshgrid")
X, Y, Z = np.mgrid[
    mins[0] : maxs[0] : 100j, mins[1] : maxs[1] : 100j, mins[2] : maxs[2] : 100j
]
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])


kernel = gaussian_kde(data_mat.T)
print("evaluating meshgrid")
Z = np.reshape(kernel(positions).T, X.shape)


# %% [markdown]
# ##
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        ax.axis("off")
        if i < j:
            sns.scatterplot(
                data=data,
                y=plot_vars[j],
                x=plot_vars[i],
                s=5,
                alpha=0.2,
                ax=ax,
                linewidth=0,
                color="black",
            )
        if i != j:
            unused = np.setdiff1d([0, 1, 2], [i, j])[0]
            ax.imshow(
                np.rot90(Z.sum(axis=unused)),  # integrate out the unused dim
                cmap=plt.cm.Reds,
                extent=[mins[i], maxs[i], mins[j], maxs[j]],
            )
            # print(Z.shape)
        ax.set_xlim([mins[i], maxs[i]])
        ax.set_ylim([mins[j], maxs[j]])
plt.tight_layout()
