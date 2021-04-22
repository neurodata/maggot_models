# %% [markdown]
# ##
import datetime
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns

from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    plot_neurons,
    plot_single_dendrogram,
    plot_volumes,
    set_axes_equal,
    set_theme,
    CLASS_COLOR_DICT,
)
from src.data import load_palette

t0 = time.time()
# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/evaluate_morphology/")

CLASS_KEY = "merge_class"
ORDER_KEY = "sum_signal_flow"
# CLUSTER_KEY = "agglom_labels_t=2.5_n_components=64"
CLUSTER_KEY = "gt_blockmodel_labels"
ORDER_ASCENDING = False
FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=save_path / "figs", format=format, dpi=300, save_on=True, **kws
    )


start_instance()


# %% load data

from src.data import load_maggot_graph

mg = load_maggot_graph()
meta = mg.nodes
meta = meta[meta["has_embedding"]].copy()

if CLASS_KEY == "merge_class":
    palette = CLASS_COLOR_DICT
else:
    palette = load_palette()

#%%
nblast_dir = Path("maggot_models/experiments/nblast/outs")

#%%
# pd.read_csv(nblast_dir / "left")

from src.nblast import preprocess_nblast

data_dir = Path("maggot_models/experiments/nblast/outs")

symmetrize_mode = "geom"
transform = "ptr"
nblast_type = "scores"

side = "left"
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)
print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")
# get neurons that are in both
left_intersect_index = np.intersect1d(meta.index, nblast_sim.index)
print(f"{len(left_intersect_index)} neurons in intersection on {side}")
# reindex appropriately
nblast_sim = nblast_sim.reindex(
    index=left_intersect_index, columns=left_intersect_index
)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)
left_sim = pd.DataFrame(data=sim, index=nblast_sim.index, columns=nblast_sim.index)