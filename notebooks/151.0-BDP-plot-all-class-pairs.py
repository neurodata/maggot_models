# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from graspy.match import GraphMatch
from graspy.plot import heatmap
from src.cluster import get_paired_inds  # TODO fix the location of this func
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.utils import invert_permutation
from src.visualization import CLASS_COLOR_DICT, adjplot

from scipy.ndimage import gaussian_filter1d

print(scipy.__version__)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


graph_type = "G"
mg = load_metagraph(graph_type, version="2020-05-26")
# mg = preprocess(
#     master_mg,
#     threshold=0,
#     sym_threshold=False,
#     remove_pdiff=True,
#     binarize=False,
#     weight="weight",
# )
meta = mg.meta


# %%
from src.visualization import stacked_barplot

labels = meta["merge_class"].values
uni_labels, counts = np.unique(labels, return_counts=True)
inds = np.argsort(-counts)

paired = meta["pair_id"] != -1

fig, axs = plt.subplots(1, 2, figsize=(20, 20))
ax = axs[0]
ax, data, uni_subcat, subcategory_colors, order, = stacked_barplot(
    labels,
    labels,
    color_dict=CLASS_COLOR_DICT,
    category_order=uni_labels[inds],
    norm_bar_width=False,
    ax=ax,
    return_data=True,
)
ax.get_legend().remove()
ax.set_title("Class membership")

ax = axs[1]
ax, data, uni_subcat, subcategory_colors, order, = stacked_barplot(
    labels,
    paired,
    category_order=uni_labels[inds],
    norm_bar_width=False,
    ax=ax,
    return_data=True,
    subcategory_order=[True, False],
    palette=sns.color_palette("tab10", 4)[2:],
)
# ax.get_legend().remove()
ax.set_title("Paired")
stashfig("classes-and-pairs")

# stashfig("all-class-bars")
