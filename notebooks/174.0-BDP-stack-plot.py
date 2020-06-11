# %% [markdown]
# ##
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from graspy.utils import binarize
from src.data import load_metagraph
from src.io import savefig
from src.visualization import set_axes_equal, stacked_barplot

# mpl.use("Qt5Agg")

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def get_square_verts(x, length):
    return [[x, 0, 0], [x, length, 0], [x, length, length], [x, 0, length], [x, 0, 0]]


mg = load_metagraph("G")
mg = mg.sort_values("merge_class")
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
adjs = []
for gt in graph_types:
    temp_mg = load_metagraph(gt)
    temp_mg = temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)


def scatter_adj_3d(A, x, scale=1, ax=None, c="grey", zorder=1):
    inds = np.nonzero(A)
    edges = A[inds]
    xs = len(edges) * [x]  # dummy variable for the "pane" x position
    ys = inds[1]  # target
    zs = inds[0]  # source
    ax.scatter(
        xs, ys, zs, s=scale * edges, c=c, zorder=zorder
    )  # note: zorder doesn't work
    return ax


# %% [markdown]
# ##


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.force_zorder = True

n_verts = len(adjs[0])
n_graphs = len(adjs)
step = n_verts

pal = sns.color_palette("deep", 4)
for i, temp_adj in enumerate(adjs):
    x = i * step - 5  # try moving the points just in front of the pane
    # note this doesn't quite work til MPL bug is fixed
    # REF: open PR https://github.com/matplotlib/matplotlib/pull/14508
    ax = scatter_adj_3d(
        binarize(temp_adj),
        x,
        ax=ax,
        scale=0.05,
        c=[pal[i]],
        zorder=n_graphs * 2 - i * 2 + 1,
    )
    vert_list = [get_square_verts(i * step, n_verts)]
    pc = Poly3DCollection(
        vert_list, edgecolors="dimgrey", facecolors="white", linewidths=1, alpha=0.85
    )
    pc.set_zorder(n_graphs * 2 - i * 2)
    ax.add_collection3d(pc)


# x will index the graphs
ax.set_xlim((-10, step * (n_graphs - 1)))
ax.set_ylim((0, n_verts))
ax.set_zlim((0, n_verts))

# set camera position
ax.azim = 135  # -45
ax.elev = 20

# # for testing
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
ylim = ax.get_ylim()
ax.set_ylim(ylim[::-1])
zlim = ax.get_zlim()
ax.set_zlim(zlim[::-1])
# xlim = ax.get_xlim()
# ax.set_xlim(xlim[::-1])
# ax.invert_zaxis()
# ax.invert_yaxis()
# ax.invert_xaxis()

pad = 0.01
label_loc = "bottom"
names = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
for i, name in enumerate(names):
    if label_loc == "top":
        ax.text(i * step, n_verts + pad * n_verts, 0 - pad * n_verts, name, ha="center")
    elif label_loc == "bottom":
        ax.text(
            i * step,
            0 - pad * n_verts,
            n_verts + pad * n_verts,
            name,
            ha="center",
            va="top",
        )
ax.axis("off")
stashfig("stacked-adj")

