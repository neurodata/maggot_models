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
from src.visualization import adjplot, CLASS_COLOR_DICT
from src.cluster import get_paired_inds

from src.io import savefig
import os
from scipy.ndimage import gaussian_filter1d

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


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
mg = mg.remove_pdiff()
mg = mg.make_lcc()
main_meta = mg.meta.copy()

graph_types = ["G", "Gad", "Gaa", "Gdd", "Gda"]
graph_names = dict(
    zip(graph_types, [r"Sum", r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"])
)
adjs = []
adj_dict = {}
mg_dict = {}
for g in graph_types:
    temp_mg = load_metagraph(g)
    temp_mg.reindex(mg.meta.index, use_ids=True)
    # temp_adj = temp_mg.adj
    # adjs.append(temp_adj)
    # adj_dict[g] = temp_adj
    if remove_missing:
        temp_mg = temp_mg.make_lcc()
    mg_dict[g] = temp_mg


graph_type_colors = dict(
    zip(graph_types, sns.color_palette("colorblind", n_colors=len(graph_types)))
)


# for mg, g in zip(adjs, graph_types):
#     sf = -signal_flow(adj)  # TODO replace with GM flow
#     meta[f"{g}_flow"] = sf
for g, mg in mg_dict.items():
    adj = mg.adj
    meta = mg.meta
    sf = -signal_flow(adj)
    meta[f"{g}_flow"] = sf
    main_meta.loc[meta.index, f"{g}_flow"] = sf


line_kws = dict(linewidth=1, linestyle="--", color="grey")


rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "grey",
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

# %% [markdown]
# ##
n_row = 8
n_col = 12
scale = 3
figsize = scale * np.array([n_col, n_row])
fig = plt.figure(figsize=figsize)  # constrained_layout=True)
gs = GridSpec(n_row, n_col, figure=fig)


# plots of the adjacency matrices


def plot_sorted_adj(graph_type, ax):
    mg = mg_dict[graph_type]
    adj = mg.adj
    meta = mg.meta
    _, _, top, _, = adjplot(
        adj,
        meta=meta,
        item_order=f"{graph_type}_flow",
        colors="merge_class",
        palette=CLASS_COLOR_DICT,
        plot_type="scattermap",
        sizes=(0.5, 1),
        ax=ax,
        color=graph_type_colors[graph_type],
    )
    top.set_title(graph_names[graph_type], color=graph_type_colors[graph_type])
    ax.plot([0, len(adj)], [0, len(adj)], **line_kws)


ax = fig.add_subplot(gs[1:3, :2])
plot_sorted_adj("G", ax)

ax = fig.add_subplot(gs[:2, 2:4])
plot_sorted_adj("Gad", ax)

ax = fig.add_subplot(gs[:2, 4:6])
plot_sorted_adj("Gaa", ax)

ax = fig.add_subplot(gs[2:4, 2:4])
plot_sorted_adj("Gdd", ax)

ax = fig.add_subplot(gs[2:4, 4:6])
plot_sorted_adj("Gda", ax)


# plots of the diagonal means


def plot_diag_vals(graph_type, ax):
    mg = mg_dict[graph_type]
    meta = mg.meta
    adj = mg.adj
    sf = meta[f"{graph_type}_flow"]
    perm_inds = np.argsort(sf)
    perm_adj = adj[np.ix_(perm_inds, perm_inds)]
    ks = np.arange(-len(adj) + 1, len(adj))
    vals = calc_mean_by_k(ks, perm_adj)
    kde_vals = gaussian_filter1d(vals, sigma=25)
    sns.scatterplot(
        x=ks,
        y=vals,
        s=10,
        alpha=0.4,
        linewidth=0,
        ax=ax,
        color=graph_type_colors[graph_type],
    )
    sns.lineplot(x=ks, y=kde_vals, ax=ax, color=graph_type_colors[graph_type])
    ax.set_xlabel("Diagonal index")
    ax.set_title(graph_names[graph_type], color=graph_type_colors[graph_type])
    ax.axvline(0, **line_kws)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))


ax = fig.add_subplot(gs[5:7, :2])
plot_diag_vals("G", ax)
ax.set_ylabel("Mean value on diagonal")

ax = fig.add_subplot(gs[4:6, 2:4])
plot_diag_vals("Gad", ax)
ax.set_xticks([])
ax.set_xlabel("")

ax = fig.add_subplot(gs[4:6, 4:6])
plot_diag_vals("Gaa", ax)
ax.set_xticks([])
ax.set_xlabel("")

ax = fig.add_subplot(gs[6:8, 2:4])
plot_diag_vals("Gdd", ax)

ax = fig.add_subplot(gs[6:8, 4:6])
plot_diag_vals("Gda", ax)


def plot_rank_scatter(g_row, g_col, ax):
    sns.scatterplot(
        data=main_meta,
        x=f"rank_{g_col}_flow",
        y=f"rank_{g_row}_flow",
        hue="merge_class",
        palette=CLASS_COLOR_DICT,
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
    ax.text(0.75, 0.05, f"{corr:.2f}", transform=ax.transAxes)


for g in graph_types:
    main_meta[f"rank_{g}_flow"] = main_meta[f"{g}_flow"].rank(ascending=False)
    mg = mg_dict[g]
    mg.meta[f"rank_{g}_flow"] = mg.meta[f"{g}_flow"].rank(ascending=False)

row_start = 0
col_start = 7
for i, g_row in enumerate(graph_types):
    for j, g_col in enumerate(graph_types):
        if i < j:
            ax = fig.add_subplot(gs[row_start + i, col_start + j])
            plot_rank_scatter(g_row, g_col, ax)
            if i == j - 1:
                ax.set_xlabel(graph_names[g_col])
                ax.xaxis.label.set_color(graph_type_colors[g_col])
                ax.set_ylabel(graph_names[g_row])
                ax.yaxis.label.set_color(graph_type_colors[g_row])

# plt.text(0.18, 0.51, "Flow ranks", ha="left", va=x"top", transform=fig.transFigure)


def calc_upper_triu(adj, method="sum"):
    triu_inds = np.triu_indices(len(adj), k=1)
    if method == "sum":
        prop = np.sum(adj[triu_inds]) / np.sum(adj)
    elif method == "fro":
        prop = np.linalg.norm(adj[triu_inds]) / np.linalg.norm(adj)
    return prop


# def shuffle_adj(adj):
#     inds = np.random.choice(len(adj), size=len(adj), replace=False)
#     adj_shuffled = adj[np.ix_(inds, inds)].copy()
#     return adj_shuffled
def shuffle_edges(A):
    fake_A = A.copy().ravel()
    np.random.shuffle(fake_A)
    fake_A = fake_A.reshape((len(A), len(A)))
    return fake_A


method = "sum"
n_shuffle = 2
rows = []
for g in graph_types:
    mg = mg_dict[g]
    adj = mg.adj
    meta = mg.meta
    sf = meta[f"{g}_flow"]
    perm_inds = np.argsort(sf)
    perm_adj = adj[np.ix_(perm_inds, perm_inds)]
    prop = calc_upper_triu(perm_adj, method=method)
    row = {"p_upper_triu": prop, "type": "True", "method": "sum", "graph_type": g}
    rows.append(row)
    for i in range(n_shuffle):
        adj_shuffled = shuffle_edges(adj)
        rand_sf = -signal_flow(adj_shuffled)
        perm_inds = np.argsort(rand_sf)
        perm_adj_shuffled = adj_shuffled[np.ix_(perm_inds, perm_inds)]
        rand_prop = calc_upper_triu(perm_adj_shuffled, method=method)
        row = {
            "p_upper_triu": rand_prop,
            "type": "Random",
            "method": "sum",
            "graph_type": g,
        }
        rows.append(row)
shuffle_df = pd.DataFrame(rows)

ax = fig.add_subplot(gs[2:4, 7:9])
ax = sns.stripplot(
    data=shuffle_df[shuffle_df["type"] == "Random"],
    x="graph_type",
    y="p_upper_triu",
    linewidth=1,
    alpha=0.4,
    jitter=0.3,
    size=5,
    ax=ax,
    hue="graph_type",
    palette=graph_type_colors,
)

ax = sns.stripplot(
    data=shuffle_df[shuffle_df["type"] == "True"],
    x="graph_type",
    y="p_upper_triu",
    marker="_",
    linewidth=2,
    s=50,
    ax=ax,
    label="True",
    jitter=False,
    hue="graph_type",
    palette=graph_type_colors,
)
ax.get_legend().remove()
ax.set_xlabel("")
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
shuffle_marker = plt.scatter([], [], marker=".", c="k", label="Shuffled")
true_marker = plt.scatter([], [], marker="_", linewidth=2, s=300, c="k", label="True")
ax.legend(handles=[shuffle_marker, true_marker])
ax.set_ylabel("Prop. synapses in upper triu")
tick_labels = []
for t in ax.get_xticklabels():
    name = t.get_text()
    tick_labels.append(graph_names[name])
    t.set_color(graph_type_colors[name])
ax.set_xticklabels(tick_labels)


# pair correlations


def plot_pair_scatter(graph_type, ax):
    mg = mg_dict[graph_type]
    meta = mg.meta
    meta["inds"] = range(len(meta))
    cols = [
        f"rank_{graph_type}_flow",
        "merge_class",
        "pair",
        "pair_id",
        "hemisphere",
        "inds",
        "left",
        "right",
    ]
    meta = meta[cols].copy()
    meta = meta[meta["pair"].isin(meta.index)]
    meta.sort_values("pair_id", inplace=True)
    meta.index.name = "skel_id"
    left_meta = meta[meta["left"]].copy()
    left_meta.reset_index(inplace=True)
    left_meta.set_index("pair_id", inplace=True)
    left_meta.rename(lambda x: "left_" + str(x), axis=1, inplace=True)
    right_meta = meta[meta["right"]].copy()
    right_meta.reset_index(inplace=True)
    right_meta.set_index("pair_id", inplace=True)
    right_meta.rename(lambda x: "right_" + str(x), axis=1, inplace=True)
    pair_meta = pd.concat((left_meta, right_meta), axis=1, ignore_index=False)
    pair_meta["merge_class"] = "unk"
    for i, r in pair_meta.iterrows():
        if r["left_merge_class"] == r["right_merge_class"]:
            pair_meta.loc[i, "merge_class"] = r["left_merge_class"]

    sns.scatterplot(
        data=pair_meta,
        x=f"left_rank_{graph_type}_flow",
        y=f"right_rank_{graph_type}_flow",
        hue="merge_class",
        palette=CLASS_COLOR_DICT,
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
    cols = [f"left_rank_{graph_type}_flow", f"right_rank_{graph_type}_flow"]
    rank_df = pair_meta[cols].copy()
    rank_df.dropna(axis=0, how="any", inplace=True)
    x_rank = rank_df[cols[0]]
    y_rank = rank_df[cols[1]]
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    ax.text(0.85, 0.05, f"{corr:.2f}", transform=ax.transAxes)
    ax.set_title(graph_names[graph_type], color=graph_type_colors[graph_type])

    return


ax = fig.add_subplot(gs[5:7, 6:8])
plot_pair_scatter("G", ax)
ax.set_xlabel("Left rank")
ax.set_ylabel("Right rank")

ax = fig.add_subplot(gs[4:6, 8:10])
plot_pair_scatter("Gad", ax)

ax = fig.add_subplot(gs[4:6, 10:12])
plot_pair_scatter("Gaa", ax)

ax = fig.add_subplot(gs[6:8, 8:10])
plot_pair_scatter("Gdd", ax)
ax.set_xlabel("Left rank")

ax = fig.add_subplot(gs[6:8, 10:12])
plot_pair_scatter("Gda", ax)
ax.set_xlabel("Left rank")


plt.tight_layout()
stashfig(f"flow-figure-remove_missing={remove_missing}")

