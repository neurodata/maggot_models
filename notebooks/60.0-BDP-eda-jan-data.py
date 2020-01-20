# %% [markdown]
# # Imports
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    sankey,
    screeplot,
    gridmap,
)


warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)

BRAIN_VERSION = "2020-01-14"

SAVEFIGS = True
SAVESKELS = False
SAVEOBJS = True


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


mg = load_metagraph("Gn", BRAIN_VERSION)
meta = mg.meta.copy()
meta["Original index"] = range(meta.shape[0])
degree_df = mg.calculate_degrees()
meta = pd.concat((meta, degree_df), axis=1)
meta.sort_values(["Hemisphere", "Class 1", "Pair ID"], inplace=True, kind="mergesort")
temp_inds = meta["Original index"]
mg = mg.reindex(temp_inds)
mg = MetaGraph(mg.adj, meta)

heatmap(
    mg.adj,
    sort_nodes=False,
    inner_hier_labels=mg["Class 1"],
    outer_hier_labels=mg["Hemisphere"],
    transform="binarize",
    cbar=False,
    figsize=(30, 30),
    hier_label_fontsize=10,
)
stashfig("heatmap")

# %% [markdown]
# #
mg.make_lcc()
adj = mg.adj
ase_latent = ase(adj, None, True)
print(f"ZG chose {ase_latent.shape[1]//2}")
n_unique = len(np.unique(mg["Class 1"]))
meta_vals = [
    "Class 1",
    "Merge Class",
    "Hemisphere",
    "is_pdiff",
    "is_usplit",
    "is_brain",
]

for meta_val in meta_vals:
    pairplot(
        ase_latent,
        labels=mg[meta_val],
        palette=cc.glasbey_light[: mg.meta[meta_val].nunique()],
        title=meta_val,
    )
    stashfig(meta_val + "-pairplot")


# %% [markdown]
# # Try saving some output
# out_path = Path("maggot_models/notebooks/outs/60.0-BDP-eda-jan-data/objs")
# save_latent = np.concatenate((ase_latent[:, :3], ase_latent[:, 4:-1]), axis=-1)
# print(save_latent.shape)
# save_latent_df = pd.DataFrame(data=save_latent)
# save_latent_df.to_csv(out_path / "save_latent.tsv", sep="\t", header=False)
# mg.meta.to_csv(out_path / "save_meta.tsv", sep="\t")
