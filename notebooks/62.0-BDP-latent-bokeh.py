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
from bokeh.embed import file_html
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FactorRange, Legend, Span, PreText
from bokeh.palettes import Spectral4, all_palettes
from bokeh.plotting import curdoc, figure, output_file, show
from bokeh.resources import CDN
from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import degreeplot, edgeplot, gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph, load_networkx
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import (
    get_blockmodel_df,
    get_sbm_prob,
    invert_permutation,
    meta_to_array,
    savefig,
)
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

from bokeh.layouts import column, row
from bokeh.models import Select
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from scipy.linalg import orthogonal_procrustes


def pair_augment(mg):
    pair_df = pd.read_csv(
        "maggot_models/data/raw/Maggot-Brain-Connectome/pairs/bp-pairs-2020-01-13_continuedAdditions.csv"
    )

    skeleton_labels = mg.meta.index.values

    # extract valid node pairings
    left_nodes = pair_df["leftid"].values
    right_nodes = pair_df["rightid"].values

    left_right_pairs = list(zip(left_nodes, right_nodes))

    left_nodes_unique, left_nodes_counts = np.unique(left_nodes, return_counts=True)
    left_duplicate_inds = np.where(left_nodes_counts >= 2)[0]
    left_duplicate_nodes = left_nodes_unique[left_duplicate_inds]

    right_nodes_unique, right_nodes_counts = np.unique(right_nodes, return_counts=True)
    right_duplicate_inds = np.where(right_nodes_counts >= 2)[0]
    right_duplicate_nodes = right_nodes_unique[right_duplicate_inds]

    left_nodes = []
    right_nodes = []
    for left, right in left_right_pairs:
        if left not in left_duplicate_nodes and right not in right_duplicate_nodes:
            if left in skeleton_labels and right in skeleton_labels:
                left_nodes.append(left)
                right_nodes.append(right)

    pair_nodelist = np.concatenate((left_nodes, right_nodes))
    not_paired = np.setdiff1d(skeleton_labels, pair_nodelist)
    sorted_nodelist = np.concatenate((pair_nodelist, not_paired))

    # sort the graph and metadata according to this
    sort_map = dict(zip(sorted_nodelist, range(len(sorted_nodelist))))
    inv_perm_inds = np.array(itemgetter(*skeleton_labels)(sort_map))
    perm_inds = invert_permutation(inv_perm_inds)

    mg.reindex(perm_inds)

    side_labels = mg["Hemisphere"]
    side_labels = side_labels.astype("<U2")
    for i, l in enumerate(side_labels):
        if mg.meta.index.values[i] in not_paired:
            side_labels[i] = "U" + l
    mg["Hemisphere"] = side_labels
    n_pairs = len(left_nodes)
    return mg, n_pairs


def max_symmetrize(mg, n_pairs):
    """ assumes that mg is sorted
    
    Parameters
    ----------
    mg : [type]
        [description]
    n_pairs : [type]
        [description]
    """
    adj = mg.adj
    left_left_adj = adj[:n_pairs, :n_pairs]
    left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
    right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
    right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

    # max, average gives similar results
    sym_ipsi_adj = np.maximum(left_left_adj, right_right_adj)
    sym_contra_adj = np.maximum(left_right_adj, right_left_adj)

    sym_adj = adj.copy()
    sym_adj[:n_pairs, :n_pairs] = sym_ipsi_adj
    sym_adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs] = sym_ipsi_adj
    sym_adj[:n_pairs, n_pairs : 2 * n_pairs] = sym_contra_adj
    sym_adj[n_pairs : 2 * n_pairs, :n_pairs] = sym_contra_adj

    sym_mg = MetaGraph(sym_adj, mg.meta)  # did not change indices order so this ok

    return sym_mg


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
BRAIN_VERSION = "2020-01-16"


# %% [markdown]
# #
thresh = 0.0
mg = load_metagraph("Gadn", BRAIN_VERSION)
mg.adj[mg.adj < thresh] = 0
mg, n_pairs = pair_augment(mg)

left_pair_ids = mg["Pair ID"][:n_pairs]
right_pair_ids = mg["Pair ID"][n_pairs : 2 * n_pairs]
print((left_pair_ids == right_pair_ids).all())

sym_mg = max_symmetrize(mg, n_pairs)
left_pair_ids = sym_mg["Pair ID"][:n_pairs]
right_pair_ids = sym_mg["Pair ID"][n_pairs : 2 * n_pairs]
print((left_pair_ids == right_pair_ids).all())

sym_mg.make_lcc()
n_pairs = sym_mg.meta["Pair ID"].nunique() - 1
left_pair_ids = sym_mg["Pair ID"][:n_pairs]
right_pair_ids = sym_mg["Pair ID"][n_pairs : 2 * n_pairs]
print((left_pair_ids == right_pair_ids).all())

uni_pair, counts = np.unique(sym_mg["Pair ID"], return_counts=True)
print(np.min(counts))

left_pair_ids = sym_mg["Pair ID"][:n_pairs]
right_pair_ids = sym_mg["Pair ID"][n_pairs : 2 * n_pairs]

latent = lse(sym_mg.adj, n_components=None)
left_latent = latent[:n_pairs, :]
right_latent = latent[n_pairs : 2 * n_pairs, :]
R, scalar = orthogonal_procrustes(left_latent, right_latent)

n_components = latent.shape[1]
class_labels = sym_mg["Class 1"]
n_unique = len(np.unique(class_labels))
sym_mg.meta["Original index"] = range(len(sym_mg.meta))
left_df = sym_mg.meta[sym_mg.meta["Hemisphere"] == "L"]
left_inds = left_df["Original index"].values
left_latent = latent[left_inds, :]
left_latent = left_latent @ R
latent[left_inds, :] = left_latent

latent_cols = [f"dim {i}" for i in range(latent.shape[1])]
latent_df = pd.DataFrame(data=latent, index=sym_mg.meta.index, columns=latent_cols)

df = pd.concat((sym_mg.meta, latent_df), axis=1)
df.index.name = "Skeleton ID"

x = "dim 0"
y = "dim 1"

color_by = "Class 1"
uni_color_classes = np.unique(df[color_by])
colormap = dict(zip(uni_color_classes, cc.glasbey_light))
colors = np.array(itemgetter(*df[color_by])(colormap))
df["colors"] = colors
source = ColumnDataSource(df)

TOOLS = "pan,reset,zoom_in,zoom_out,box_zoom,lasso_select"

scatter_tooltips = [
    ("ID", "@{Skeleton ID}"),
    ("Class 1", "@{Class 1}"),
    ("Class 2", "@{Class 2}"),
    ("Side", "@Hemisphere"),
    ("Name", "@{ neuron_name}"),
]

stats = PreText(text="HERE", width=500)


def create_figure():

    scatter_fig = figure(
        tools=TOOLS,
        plot_width=1000,
        plot_height=800,
        min_border=10,
        min_border_left=50,
        toolbar_location="above",
        x_axis_location=None,
        y_axis_location=None,
        tooltips=scatter_tooltips,
    )
    scatter_fig.xgrid.visible = False
    scatter_fig.ygrid.visible = False

    legend_it = []
    for c in uni_color_classes:
        temp_df = df[df[color_by] == c]
        source = ColumnDataSource(temp_df)
        out = scatter_fig.circle(
            x=x.value,
            y=y.value,
            source=source,
            alpha=0.6,
            line_color=None,
            fill_color="colors",
            # legend_group=color_by,
        )
        legend_it.append((c, [out]))

    legend = Legend(items=legend_it, location=(0, 1.5))
    legend.click_policy = "hide"
    scatter_fig.add_layout(legend, "right")

    selected = source.selected.indices
    print(selected)
    # if selected:

    return scatter_fig


def update(attr, old, new):
    layout.children[0] = create_figure()
    update_stats(source.data)


def update_stats(data):
    stats.text = str(data)


def selection_change(attr, old, new):
    selected = source.selected.indices
    data = source.data
    if selected:
        update_stats(data.iloc[selected, :3])


x = Select(title="X-Axis", value="dim 0", options=latent_cols)
x.on_change("value", update)

y = Select(title="Y-Axis", value="dim 1", options=latent_cols)
y.on_change("value", update)

source.selected.on_change("indices", selection_change)


controls = column(x, y, width=200)
layout = row(create_figure(), controls, stats)

curdoc().add_root(layout)
curdoc().title = BRAIN_VERSION
# show(scatter_fig)
show(layout)
# html = file_html(scatter_fig, CDN, "mb_adj")
# with open("full_new_adj.html", "w") as f:
#     f.write(html)


# %%
