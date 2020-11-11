# %% [markdown]
# ##
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_metagraph
from src.hierarchy import signal_flow
from src.visualization import adjplot, CLASS_COLOR_DICT, set_theme


from src.io import savefig
import os
from scipy.ndimage import gaussian_filter1d

FNAME = os.path.basename(__file__)[:-3]
rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    # "axes.edgecolor": "lightgrey",
    # "ytick.color": "grey",
    # "xtick.color": "grey",
    # "axes.labelcolor": "dimgrey",
    # "text.color": "dimgrey",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}

set_theme(rc_dict=rc_dict, font_scale=1.25)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def diag_indices(length, k=0):
    neg = False
    if k < 0:
        neg = True
    k = np.abs(k)
    inds = (np.arange(length - k), np.arange(k, length))
    if neg:
        return (inds[1], inds[0])
    else:
        return inds


def calc_mean_by_k(ks, perm_adj):
    length = len(perm_adj)
    ps = []
    for k in ks:
        p = perm_adj[diag_indices(length, k)].mean()
        ps.append(p)
    return np.array(ps)


def get_vals_by_k(ks, perm_adj):
    ys = []
    xs = []
    for k in ks:
        y = perm_adj[diag_indices(len(perm_adj), k)]
        ys.append(y)
        x = np.full(len(y), k)
        xs.append(x)
    return np.concatenate(ys), np.concatenate(xs)


remove_missing = True

mg = load_metagraph("G")
# mg = mg.remove_pdiff()
# mg = mg.make_lcc()
# main_meta = mg.meta.copy()

graph_type = "Gad"
n_init = 256
max_hops = 16
allow_loops = False
walk_spec = f"gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}"
meta_path = f"maggot_models/experiments/walk_sort/outs/meta_w_order-{walk_spec}.csv"
walk_meta = pd.read_csv(meta_path, index_col=0)
# walk_meta = walk_meta.reindex(mg.meta.index)
walk_meta = walk_meta.sort_values("median_node_visits", ascending=True)

print(len(walk_meta))
unvisit_meta = walk_meta[walk_meta["median_node_visits"].isna()]
print(len(unvisit_meta))

#%%
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
graph_names = dict(
    zip(graph_types, [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"])
)
adjs = []
adj_dict = {}
mg_dict = {}
for g in graph_types:
    temp_mg = load_metagraph(g)
    if remove_missing:
        temp_mg = temp_mg.make_lcc()
    temp_walk_meta = walk_meta.reindex(index=temp_mg.meta.index.values)
    temp_walk_meta = temp_walk_meta.sort_values("median_node_visits", ascending=True)
    temp_walk_meta = temp_walk_meta[
        ~temp_walk_meta["median_node_visits"].isna()
    ]  # HACK
    temp_mg.reindex(temp_walk_meta.index, use_ids=True, inplace=True)
    temp_mg.meta["median_node_visits"] = temp_walk_meta["median_node_visits"]
    mg_dict[g] = temp_mg

colors = sns.color_palette("deep", n_colors=len(graph_types))
graph_type_colors = dict(zip(graph_types, colors))
# graph_type_colors[graph_types[0]] = colors[-1]

# %% [markdown]
# ##

line_kws = dict(linewidth=1, linestyle="--", color="grey")


def plot_sorted_adj(graph_type, ax):
    mg = mg_dict[graph_type]
    adj = mg.adj
    meta = mg.meta
    _, _, top, _, = adjplot(
        adj,
        meta=meta,
        # item_order=f"median_node_visits",
        colors="merge_class",
        palette=CLASS_COLOR_DICT,
        plot_type="scattermap",
        sizes=(0.5, 1),
        ax=ax,
        color=graph_type_colors[graph_type],
    )
    top.set_title(
        graph_names[graph_type], color=graph_type_colors[graph_type], fontsize="x-large"
    )
    ax.plot([0, len(adj)], [0, len(adj)], **line_kws)


def plot_diag_vals(graph_type, ax, mode="values", sigma=25):
    mg = mg_dict[graph_type]
    adj = mg.adj
    ks = np.arange(-len(adj) + 1, len(adj))
    vals = calc_mean_by_k(ks, adj)
    if mode == "values":
        sns.scatterplot(
            x=ks,
            y=vals,
            s=10,
            alpha=0.4,
            linewidth=0,
            ax=ax,
            color=graph_type_colors[graph_type],
        )
    elif mode == "kde":
        kde_vals = gaussian_filter1d(vals, sigma=sigma, mode="constant")
        sns.lineplot(x=ks, y=kde_vals, ax=ax, color=graph_type_colors[graph_type])
    upper_mass = adj[np.triu_indices_from(adj, k=1)].mean()
    lower_mass = adj[np.tril_indices_from(adj, k=1)].mean()
    upper_mass_prop = upper_mass / (upper_mass + lower_mass)
    lower_mass_prop = lower_mass / (upper_mass + lower_mass)
    upper_text = f"{upper_mass_prop:.2f}"
    lower_text = f"{lower_mass_prop:.2f}"
    ax.text(0.1, 0.8, lower_text, transform=ax.transAxes, color="black")
    ax.text(
        0.9,
        0.8,
        upper_text,
        ha="right",
        transform=ax.transAxes,
        color="black",
    )
    # ax.set_title(graph_names[graph_type], color=graph_type_colors[graph_type])
    ax.axvline(0, **line_kws)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.NullLocator())
    # ax.xaxis.set_major_locator(plt.NullLocator())


fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for i, graph_type in enumerate(graph_types):
    plot_sorted_adj(graph_type, axs[0, i])
    # plot_diag_vals(graph_type, axs[1, i], mode="values")
    # plot_diag_vals(graph_type, axs[2, i], mode="kde")
    plot_diag_vals(graph_type, axs[1, i], mode="kde", sigma=50)
# axs[3, 1].set_xlabel("Diagonal index", x=1.5)

ax = axs[1, 0]
ax.text(0.1, 0.9, r"$p$ back", transform=ax.transAxes, color="black")
ax.text(0.9, 0.9, r"$p$ fwd", transform=ax.transAxes, color="black", ha="right")
fig.text(0.47, 0, "Diagonal index")
axs[1, 0].set_ylabel("Mean synapse mass")
# axs[1, 0].set_xticklabels([])
# axs[2, 0].set_ylabel(r"Mean synapse mass (smoothed, $\sigma = 25$)")
# axs[3, 0].set_ylabel(r"Mean synapse mass (smoothed, $\sigma = 50$)")
# plt.tight_layout()
stashfig(f"adj-row-sort-by-walkspec-{walk_spec}")
