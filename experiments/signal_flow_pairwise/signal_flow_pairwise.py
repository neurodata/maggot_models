# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data import load_maggot_graph
from src.hierarchy import signal_flow
from src.data import load_palette

from src.io import savefig
from pathlib import Path

from src.visualization import set_theme


set_theme()


save_path = Path("maggot_models/experiments/signal_flow_pairwise/")


def stashfig(name, **kws):
    savefig(name, pathname=save_path / "figs", save_on=True, **kws)
    savefig(name, pathname=save_path / "figs", save_on=True, format="pdf", **kws)


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
mg.to_largest_connected_component()
main_meta = mg.nodes.copy()


graph_types = ["sum", "ad", "aa", "dd", "da"]
graph_names = dict(
    zip(graph_types, [r"Sum", r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"])
)
adjs = []
adj_dict = {}
mg_dict = {}
for g in graph_types:
    type_mg = mg.to_edge_type_graph(g)
    if remove_missing:
        type_mg.to_largest_connected_component()
    mg_dict[g] = type_mg

colors = sns.color_palette("deep", n_colors=len(graph_types))
graph_type_colors = dict(zip(graph_types[1:], colors))
graph_type_colors[graph_types[0]] = colors[-1]

#%%
for g, mg in mg_dict.items():
    adj = mg.adj
    meta = mg.nodes
    sf = -signal_flow(adj)
    meta[f"{g}_flow"] = sf
    main_meta.loc[meta.index, f"{g}_flow"] = sf


#%%

node_palette = load_palette()


def plot_rank_scatter(g_row, g_col, ax):
    sns.scatterplot(
        data=main_meta,
        x=f"rank_{g_col}_flow",
        y=f"rank_{g_row}_flow",
        hue="simple_group",
        palette=node_palette,
        linewidth=0,
        alpha=0.5,
        s=10,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    cols = [f"rank_{g_col}_flow", f"rank_{g_row}_flow"]
    rank_df = main_meta[cols].copy()
    rank_df.dropna(axis=0, how="any", inplace=True)
    x_rank = rank_df[cols[0]]
    y_rank = rank_df[cols[1]]
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    ax.text(
        0.75,
        0.05,
        f"{corr:.2f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0),
    )


for g in graph_types:
    main_meta[f"rank_{g}_flow"] = main_meta[f"{g}_flow"].rank(ascending=False)
    mg = mg_dict[g]
    mg.nodes[f"rank_{g}_flow"] = mg.nodes[f"{g}_flow"].rank(ascending=False)

fig, axs = plt.subplots(5, 5, figsize=(15, 15))

for i, g_row in enumerate(graph_types):
    for j, g_col in enumerate(graph_types):
        ax = axs[i, j]
        if i < j:
            plot_rank_scatter(g_row, g_col, ax)
            # if i == j - 1:
            ax.set_xlabel(graph_names[g_col])
            ax.xaxis.label.set_color(graph_type_colors[g_col])
            ax.set_ylabel(graph_names[g_row])
            ax.yaxis.label.set_color(graph_type_colors[g_row])
        else:
            ax.axis("off")

# stashfig("pairwise-rank-signal-flow")
