#%%
import os
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.plot import gridplot, heatmap
from src.data import load_metagraph
from src.io import savefig
from src.visualization import gridmap, remove_spines

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-01-21"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


# %% [markdown]
# # Load the data

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
mgs = []

for gt in graph_types:
    mg = load_metagraph(gt, version=BRAIN_VERSION)
    mgs.append(mg)

# %% [markdown]
# # Describe the data, just plot adjacency matrices

sns.set_context("talk", font_scale=2)
graph_names = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
palette = sns.color_palette("deep", 4)
fig, axs = plt.subplots(2, 2, figsize=(30, 30))  # harex=True, sharey=True)
axs = axs.ravel()
for i, mg in enumerate(mgs):
    ax = axs[i]
    mg.sort_values(["Hemisphere", "dendrite_input"], ascending=False)
    meta = mg.meta
    meta["Original index"] = range(len(meta))
    first_df = mg.meta.groupby(["Hemisphere"]).first()
    first_inds = list(first_df["Original index"].values)
    first_inds.append(len(meta) + 1)
    middle_df = mg.meta.groupby(["Hemisphere"]).mean()
    middle_inds = list(middle_df["Original index"].values)
    middle_labels = list(middle_df.index.values)
    gridmap(mg.adj, ax=ax, sizes=(8, 16), color=palette[i])
    remove_spines(ax)
    ax.set_xticks(middle_inds)
    ax.set_xticklabels(middle_labels)
    ax.set_yticks(middle_inds)
    ax.set_yticklabels(middle_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim((-2, len(meta) + 2))
    ax.set_ylim((len(meta) + 2, -2))
    for t in first_inds:
        ax.axhline(t - 0.5, 0.02, 0.98, color="grey", linestyle="--", alpha=0.5)
        ax.axvline(t - 0.5, 0, len(meta), color="grey", linestyle="--", alpha=0.5)

    n_edge = np.count_nonzero(mg.adj)
    n_syn = np.sum(mg.adj)

    ax.set_title(graph_names[i] + f" ({n_edge} edges, {n_syn:.0f} synapses)")


plt.tight_layout()
axs[0].set_xticks([])
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[3].set_yticks([])
stashfig("4-color-split-gridmap-hemisphere")

# %% [markdown]
# # Plot the adjacency matrices with class labels


class_labels = mgs[0].meta["Class 1"].copy()
uni_labels, counts = np.unique(class_labels, return_counts=True)
count_thresh = 20
counts[counts < count_thresh] = -1

is_low_labels = np.zeros(len(uni_labels), dtype=bool)
is_low_labels[counts == -1] = True

label_map = dict(zip(uni_labels, is_low_labels))

is_low_class = np.array(itemgetter(*class_labels)(label_map))
class_labels[is_low_class] = "Other"
mg.meta["Simple class"] = class_labels

gridplot(
    [mg.adj for mg in mgs],
    labels=graph_names,
    sizes=(10, 15),
    height=20,
    hier_label_fontsize=15,
    inner_hier_labels=mg["Simple class"],
    palette="deep",
    # legend=False,
)
stashfig("4-color-gridplot-class")
