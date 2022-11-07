#%%
import pymaid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from src.data import load_maggot_graph, load_palette
from src.io import savefig
from src.visualization import adjplot, set_theme

set_theme(font_scale=1.25)


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/flow_row/figs",
        **kws,
    )
    savefig(
        name,
        pathname="./maggot_models/experiments/flow_row/figs",
        format="pdf",
        **kws,
    )


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
mg = load_maggot_graph()

# %%
graph_types = ["ad", "aa", "dd", "da"]
graph_names = dict(
    zip(graph_types, [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"])
)
colors = sns.color_palette("deep", n_colors=len(graph_types))
graph_type_colors = dict(zip(graph_types, colors))

#%%
nodes = mg.nodes.copy()
nodes = nodes[~nodes["sum_rank_signal_flow"].isna()]
nodes.sort_values("sum_rank_signal_flow", inplace=True)
nodes["order"] = range(len(nodes))
mg = mg.node_subgraph(nodes.index)
mg.nodes = nodes


#%%

line_kws = dict(linewidth=1, linestyle="--", color="grey")

palette = load_palette()


def plot_sorted_adj(graph_type, ax):
    adj = mg.to_edge_type_graph(graph_type).adj
    meta = mg.nodes
    _, _, top, _, = adjplot(
        adj,
        meta=meta,
        item_order="order",
        colors="simple_group",
        palette=palette,
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
    adj = mg.to_edge_type_graph(graph_type).adj
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
    ax.axvline(0, **line_kws)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_xticks([-3000, 0, 3000])
    ax.set_xticklabels([-3000, 0, 3000])


fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for i, graph_type in enumerate(graph_types):
    plot_sorted_adj(graph_type, axs[0, i])
    plot_diag_vals(graph_type, axs[1, i], mode="kde", sigma=75)

ax = axs[1, 0]
ax.text(0.1, 0.9, r"$p$ back", transform=ax.transAxes, color="black")
ax.text(0.9, 0.9, r"$p$ fwd", transform=ax.transAxes, color="black", ha="right")
fig.text(0.47, 0.05, "Distance in sorting")
axs[1, 0].set_ylabel("Mean synapse mass")
stashfig("adj-row-sort-by-sf")

#%%
# from graspologic.match import GraphMatch

# adj = mg.sum.adj
# # constructing the match matrix
# match_mat = np.zeros_like(adj)
# triu_inds = np.triu_indices(len(match_mat), k=1)
# match_mat[triu_inds] = 1

# # running graph matching
# np.random.seed(8888)
# gm = GraphMatch(n_init=1, max_iter=100, eps=1e-6)
# gm.fit(match_mat, adj)
# perm_inds = gm.perm_inds_

# adj_matched = adj[perm_inds][:, perm_inds]
# upsets = adj_matched[triu_inds[::-1]].sum()
# upset_ration = upsets / adj_matched.sum()
