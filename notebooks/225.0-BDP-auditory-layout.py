#%% [markdown]
# # Layouts for an auditory connectome
#%%
import datetime
import time
from pathlib import Path

import colorcet as cc
import h5py
import numpy as np
import pandas as pd
from giskard.plot import graphplot
from graspologic.layouts.colors import _get_colors
from matplotlib.lines import Line2D
from src.io import savefig

t0 = time.time()


def stashfig(name):
    savefig(name, foldername="auditory-layout", pad_inches=0.05)


#%% [markdown]
# ## Load data
#%%
mat_path = Path("maggot_models/data/raw/allAudNeurons_connMat_noDupSyn.mat")
f = h5py.File(mat_path)
adj_key = "cxns_noDup"
label_key = "neuronClasses"

adj = np.array(f[adj_key])
adj[np.arange(len(adj)), np.arange(len(adj))] = 0
references = np.array(f[label_key][0])
coded_labels = [f[ref] for ref in references]
labels = np.array(["".join(chr(c[0]) for c in cl) for cl in coded_labels])

meta = pd.DataFrame(index=np.arange(len(labels)))
meta["labels"] = labels

#%% [markdown]
# ## Generate Layouts
#%%

main_random_state = np.random.default_rng(8888)


def make_legend(palette, ax, s=5):
    elements = []
    legend_labels = []
    for label, color in palette.items():
        element = Line2D(
            [0], [0], marker="o", lw=0, label=label, color=color, markersize=s
        )
        legend_labels.append(label)
        elements.append(element)
    ax.legend(
        handles=elements,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=6,
    )


def make_palette(cmap="thematic", random_state=None):
    if random_state is None:
        random_state = np.random.default_rng()
    if cmap == "thematic":
        colors = _get_colors(True, None)["nominal"]
    if cmap == "glasbey":
        colors = cc.glasbey_light.copy()
        random_state.shuffle(colors)
    palette = dict(zip(np.unique(labels), colors))
    return palette


n_repeats = 1
for i in range(n_repeats):
    random_seed = main_random_state.integers(np.iinfo(np.int32).max)
    for cmap in ["thematic"]:
        random_state = np.random.default_rng(random_seed)
        palette = make_palette(cmap, main_random_state)
        ax = graphplot(
            adj,
            n_components=32,
            n_neighbors=32,
            embedding_algorithm="ase",
            meta=meta,
            hue="labels",
            palette=palette,
            sizes=(20, 90),
            network_order=2,
            normalize_power=True,
            random_state=random_state,
            supervised_weight=0.2,
        )
        make_legend(palette, ax)
        stashfig(f"auditory-layout-seed={random_seed}-cmap={cmap}")

#%% [markdown]
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
