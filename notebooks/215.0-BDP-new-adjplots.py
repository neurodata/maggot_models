# %% [markdown]
# ##
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from anytree import Node, NodeMixin, Walker
from graspy.embed import OmnibusEmbed, selectSVD
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)
from scipy.stats import poisson
from topologic.io import tensor_projection_writer

from src.cluster import BinaryCluster
from src.data import load_metagraph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
    set_theme,
)


# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)


CLASS_KEY = "simple_class"  # "merge_class"
group_order = "median_node_visits"
FORMAT = "pdf"


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


# %% [markdown]
# ##
omni_method = "color_iso"
d = 8
bic_ratio = 0.95
min_split = 32

basename = f"-method={omni_method}-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
meta = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/meta{basename}.csv",
    index_col=0,
)
meta["lvl0_labels"] = meta["lvl0_labels"].astype(str)
adj_df = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/adj{basename}.csv",
    index_col=0,
)
adj = adj_df.values

name_map = {
    "Sens": "Sensory",
    "LN": "Local",
    "PN": "Projection",
    "KC": "Kenyon cell",
    "LHN": "Lateral horn",
    "MBIN": "MBIN",
    "Sens2o": "2nd order sensory",
    "unk": "Unknown",
    "MBON": "MBON",
    "FBN": "MB feedback",
    "CN": "Convergence",
    "PreO": "Pre-output",
    "Outs": "Output",
    "Motr": "Motor",
}
meta["simple_class"] = meta["simple_class"].map(name_map)
print(meta["simple_class"].unique())
# meta["merge_class"] = meta["simple_class"]  # HACK


graph_type = "Gad"
n_init = 256
max_hops = 16
allow_loops = False
include_reverse = False
walk_spec = f"gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}"
walk_meta = pd.read_csv(
    f"maggot_models/experiments/walk_sort/outs/meta_w_order-{walk_spec}-include_reverse={include_reverse}.csv",
    index_col=0,
)
meta["median_node_visits"] = walk_meta["median_node_visits"]  # make the sorting right


lowest_level = 7
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]

#%%


def sort_meta(meta, group, group_order=group_order, item_order=[], ascending=True):
    sort_class = group
    group_order = [group_order]
    total_sort_by = []
    for sc in sort_class:
        for co in group_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    meta = meta.sort_values(total_sort_by, ascending=ascending)
    return meta


sorted_meta = meta.copy()
sorted_meta["sort_inds"] = np.arange(len(sorted_meta))
group = level_names + ["merge_class"]
sorted_meta = sort_meta(
    sorted_meta,
    group,
    group_order=group_order,
    item_order=["merge_class", "median_node_visits"],
)
sorted_meta["new_inds"] = np.arange(len(sorted_meta))
sorted_meta[["merge_class", "lvl7_labels", "median_node_visits"]]

sort_inds = sorted_meta["sort_inds"]
sorted_adj = adj[np.ix_(sort_inds, sort_inds)]


cut = None  # where to draw a dashed line on the dendrogram
line_level = 6
level = lowest_level

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax, divider, top, _ = adjplot(
    sorted_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    sort_class=group[:line_level],
    item_order="new_inds",
    class_order=group_order,
    meta=sorted_meta,
    palette=CLASS_COLOR_DICT,
    colors=CLASS_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),  # 0.2
    dendrogram=7,
)

#%%

