#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_paired_inds
from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.utils import binarize, remove_loops
from scipy.stats import poisson
from src.data import join_node_meta, load_maggot_graph, load_palette
from src.io import savefig
from src.visualization import set_theme
from src.visualization import adjplot

t0 = time.time()
set_theme()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/plot_clustering/figs",
        **kws,
    )


mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
nodes = mg.nodes
adj = mg.sum.adj

CLUSTER_KEY = "co_cluster_n_clusters=85"
HUE_KEY = "simple_group"
palette = load_palette()

adjplot(
    adj,
    meta=nodes,
    plot_type="scattermap",
    sort_class=CLUSTER_KEY,
    class_order="sum_walk_sort",
    item_order=HUE_KEY,
    ticks=False,
    colors=HUE_KEY,
    palette=palette,
    sizes=(1, 2),
    gridline_kws=dict(linewidth=0.5, linestyle=":", color="grey"),
)
stashfig(f"adjacency-matrix-cluster_key={CLUSTER_KEY}")
