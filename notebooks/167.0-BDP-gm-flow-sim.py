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
from sklearn.metrics import pairwise_distances

from graspy.embed import ClassicalMDS
from graspy.match import GraphMatch
from graspy.plot import heatmap
from graspy.simulations import sbm, sbm_corr
from src.data import load_metagraph
from src.graph import preprocess
from src.io import savecsv, savefig
from src.utils import invert_permutation
from src.visualization import CLASS_COLOR_DICT, adjplot

print(scipy.__version__)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


def random_permutation(n):
    perm_inds = np.random.choice(int(n), replace=False, size=int(n))
    P = np.zeros((n, n))
    P[np.arange(len(P)), perm_inds] = 1
    return P


# %% [markdown]
# ##

show_adjs = False
B = np.array([[0.3, 0.7], [0.05, 0.3]])
rho = 0.9
n_per_block = 10
n_blocks = len(B)
comm = n_blocks * [n_per_block]

n_init = 50
eps = 1.0
n_samples = 20

rows = []
for i in range(n_samples):
    A1, A2 = sbm_corr(comm, B, rho)
    max_score = np.trace(A1 @ A2.T)

    shuffle_inds = np.random.choice(len(A1), replace=False, size=len(A1))
    A2_shuffle = A2[np.ix_(shuffle_inds, shuffle_inds)]

    if show_adjs:
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        heatmap(A1, ax=axs[0], cbar=False, title="Graph 1")
        heatmap(A2, ax=axs[1], cbar=False, title="Graph 2")
        heatmap(A1 - A2, ax=axs[2], cbar=False, title="Diff (G1 - G2)")
        heatmap(A2_shuffle, ax=axs[3], cbar=False, title="Graph 2 shuffled")

        P = np.zeros_like(A1)
        P[np.arange(len(P)), shuffle_inds] = 1
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        heatmap(A2, ax=axs[0], cbar=False, title="Graph 2")
        heatmap(P @ A2 @ P.T, ax=axs[1], cbar=False, title="P shuffled")
        heatmap(A2_shuffle, ax=axs[2], cbar=False, title="Index shuffled")
        heatmap(P.T @ A2_shuffle @ P, ax=axs[3], cbar=False, title="P unshuffled")

    for init_weight in [0, 0.25, 0.5, 0.75, 0.9, 0.95, "random"]:

        import matplotlib.transforms as transforms

        n_verts = A1.shape[0]

        all_positions = []
        init_indicator = []

        gm = GraphMatch(
            n_init=n_init,
            init="barycenter",
            init_weight=init_weight,
            max_iter=20,
            shuffle_input=False,
            eps=eps,
        )
        gm.fit(A1, A2_shuffle)
        results = gm.results_
        progress = gm.progress_
        final_scores = progress.groupby("init_idx")["pseudoscore"].max()
        progress["final_score"] = progress["init_idx"].map(final_scores)
        progress["optimal"] = np.abs((progress["final_score"] - max_score)) < 0.1
        p_found = progress["optimal"].mean()
        row = {"p_found": p_found, "init_weight": init_weight}
        rows.append(row)

init_weight_results = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=init_weight_results, x="init_weight", y="p_found", ax=ax)
mean_results = init_weight_results.groupby("init_weight").mean().reset_index()
sns.stripplot(
    data=mean_results,
    x="init_weight",
    y="p_found",
    ax=ax,
    size=30,
    marker="_",
    linewidth=1,
)

# %% [markdown]
# ##

fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax = axs[0]
trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
sns.lineplot(
    data=progress,
    x="iter",
    y="grad",
    hue="init_idx",
    palette="tab20",
    legend=False,
    alpha=0.5,
    linewidth=1,
    ax=ax,
)
sns.lineplot(
    data=progress[progress["optimal"]],
    x="iter",
    y="grad",
    hue="init_idx",
    palette="tab20",
    legend=False,
    alpha=1,
    ax=ax,
)
ax.axhline(eps, linestyle=":", color="darkred")
ax.text(1.01, eps, "Epsilon", transform=trans, va="center", color="darkred")
ax.set_ylabel("Gradient")

ax = axs[1]
trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
sns.lineplot(
    data=progress,
    x="iter",
    y="pseudoscore",
    hue="init_idx",
    palette="tab20",
    legend=False,
    alpha=0.5,
    ax=ax,
    linewidth=1,
)
sns.lineplot(
    data=progress[progress["optimal"]],
    x="iter",
    y="pseudoscore",
    hue="init_idx",
    palette="tab20",
    legend=False,
    alpha=1,
    ax=ax,
)
ax.axhline(max_score, linestyle=":", color="black")

ax.text(
    1.01,
    max_score,
    f"Optimal\n{p_max:0.2f} found",
    transform=trans,
    va="center",
    color="black",
)
ax.set_ylabel("Score")
ax.set_xlabel("Iteration")
stashfig("score-by-iter")


# %% [markdown]
# ##

from src.flow import make_exp_match, fit_gm_exp, diag_indices


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


low_p = 0.01
diag_p = 0.1
feedforward_p = 0.3
n_blocks = 10
n_per_block = 25  # 50
community_sizes = n_blocks * [n_per_block]

basename = f"-n_blocks={n_blocks}-n_per_block={n_per_block}"

block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=axs[0], square=True)
axs[0].xaxis.tick_top()
axs[0].set_title("Block probability matrix", pad=25)

np.random.seed(88)
adj, labels = sbm(
    community_sizes, block_probs, directed=True, loops=False, return_labels=True
)
n_verts = adj.shape[0]

adjplot(adj, sort_class=labels, cbar=False, ax=axs[1], square=True)
axs[1].set_title("Adjacency matrix", pad=25)
plt.tight_layout()
stashfig("sbm" + basename)

# %% [markdown]
# ##

currtime = time.time()

n_verts = len(adj)

halfs = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

alphas = [np.round(np.log(2) / (h * n_verts), decimals=7) for h in halfs]

from sklearn.model_selection import ParameterGrid

param_grid = {
    "alpha": alphas[:4],
    "beta": [1, 0.5, 0.3],  # 0.9, 0.7, 0.5, 0.3, 0.1],
    "norm": [False],
    "c": [0],
}
params = list(ParameterGrid(param_grid))


def calc_accuracy(block_preds):
    acc = (block_preds == labels).astype(float).mean()
    return acc


def calc_abs_dist(block_preds):
    mae = np.abs(block_preds - labels).mean()
    return mae


def calc_euc_dist(block_preds):
    sse = np.sqrt(((block_preds - labels) ** 2).sum())
    mse = sse / len(block_preds)
    return mse


def calc_scores(perm):
    block_preds = perm // n_per_block
    acc = calc_accuracy(block_preds)
    mae = calc_abs_dist(block_preds)
    mse = calc_euc_dist(block_preds)
    return acc, mae, mse
    # ax.text(
    #     0.75,
    #     0.07,
    #     f"Acc. {acc:.2f}\nMAE {mae:.2f}\nMSE {mse:.2f}",
    #     transform=ax.transAxes,
    # )


n_init = 25

rows = []
for p in params:
    gm = GraphMatch(
        n_init=n_init, init="barycenter", init_weight=0.9, max_iter=20, eps=1
    )
    match = make_exp_match(adj, **p)
    gm.fit(adj, match)
    perm = gm.perm_inds_
    acc, mae, mse = calc_scores(perm)
    row = p.copy()
    row["acc"] = acc
    row["mae"] = mae
    row["mse"] = mse
    row["score"] = gm.score_
    row["match_sum"] = np.sum(match)
    row["match_fro"] = np.linalg.norm(match)
    rows.append(row)

# %% [markdown]
# ##
res_df = pd.DataFrame(rows)
res_df["norm_score"] = res_df["score"] / res_df

heatmap_kws = dict(annot=True, annot_kws={"size": 8}, cmap="Reds")
fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
ax = axs[0]
score_df = res_df.pivot(index="alpha", columns="beta", values="score")
sns.heatmap(data=score_df, ax=ax, **heatmap_kws)

ax = axs[1]
acc_df = res_df.pivot(index="alpha", columns="beta", values="acc")
sns.heatmap(data=acc_df, ax=ax, **heatmap_kws)

plt.yticks(rotation=0)
# %% [markdown]
# ##
fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
ax = axs[0]
unnorm_df = res_df[res_df["norm"] == False]
corr_df = unnorm_df.pivot(index="alpha", columns="beta", values="corr")
sns.heatmap(data=corr_df, ax=ax, **heatmap_kws)
ax.set_title("Norm = False")
ax = axs[1]
fro_df = res_df[res_df["norm"] == "fro"]
corr_df = fro_df.pivot(index="alpha", columns="beta", values="corr")
sns.heatmap(data=corr_df, ax=ax, **heatmap_kws)
ax.set_title("Norm = fro")
ax = axs[2]
sum_df = res_df[res_df["norm"] == "sum"]
corr_df = sum_df.pivot(index="alpha", columns="beta", values="corr")
sns.heatmap(data=corr_df, ax=ax, **heatmap_kws)
ax.set_title("Norm = sum")
plt.yticks(rotation=0)
stashfig("corr-heatmaps" + basename)

fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
ax = axs[0]
unnorm_df = res_df[res_df["norm"] == False]
corr_df = unnorm_df.pivot(index="alpha", columns="beta", values="score")
sns.heatmap(data=corr_df, ax=ax, **heatmap_kws)
ax.set_title("Norm = False")
ax = axs[1]
fro_df = res_df[res_df["norm"] == "fro"]
corr_df = fro_df.pivot(index="alpha", columns="beta", values="score")
sns.heatmap(data=corr_df, ax=ax, **heatmap_kws)
ax.set_title("Norm = fro")
ax = axs[2]
sum_df = res_df[res_df["norm"] == "sum"]
corr_df = sum_df.pivot(index="alpha", columns="beta", values="score")
sns.heatmap(data=corr_df, ax=ax, **heatmap_kws)
ax.set_title("Norm = sum")
plt.yticks(rotation=0)
stashfig("score-heatmaps" + basename)


# %% [markdown]
# ##
# indicator = np.full(len(gm.positions_), i)
# all_positions += gm.positions_
# init_indicator.append(indicator)

init_indicator.append(["Barycenter"])
init_indicator.append(["Truth"])
init_indicator = np.concatenate(init_indicator)
# init_indicator = np.array(init_indicator)
all_positions.append(np.full(A1.shape, 1 / A1.size))
all_positions.append(P.T)
all_positions = np.array(all_positions)
all_positions = all_positions.reshape((len(all_positions), -1))


position_pdist = pairwise_distances(all_positions, metric="euclidean")


cmds = ClassicalMDS(n_components=2, dissimilarity="euclidean")
all_X = cmds.fit_transform(all_positions)
all_X -= all_X[-1]

# remove_rand = False
# if remove_rand:
#     X = all_X[n_rand:]
#     init_indicator = init_indicator[n_rand:]
# else:
X = all_X


plot_df = pd.DataFrame(data=X)
plot_df["init"] = init_indicator
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# sns.scatterplot(data=plot_df[plot_df["init"] == "Random"], x=0, y=1, ax=ax)
sns.lineplot(
    data=plot_df[~plot_df["init"].isin(["Barycenter", "Truth", "Random"])],
    x=0,
    y=1,
    hue="init",
    palette=sns.color_palette("husl", n_init),
    ax=ax,
    legend=False,
    # markers=True,
    # style="init",
)
sns.scatterplot(
    data=plot_df[plot_df["init"] == "Barycenter"],
    x=0,
    y=1,
    ax=ax,
    s=200,
    marker="s",
    color="slategrey",
)
sns.scatterplot(
    data=plot_df[plot_df["init"] == "Truth"],
    x=0,
    y=1,
    ax=ax,
    s=400,
    marker="*",
    color="green",
    alpha=0.8,
)
collections = ax.collections
collections[-1].set_zorder(n_init + 100)
collections[-2].set_zorder(n_init + 200)
ax.axis("off")

# %%
n_rand = 100
permutations = [random_permutation(n_verts) for _ in range(n_rand)]
random_stochastics = [random_permutation(n_verts) for _ in range(n_rand)]
barycenter = np.full(A1.shape, 1 / A1.size)
all_positions = []
all_positions += permutations
all_positions += random_stochastics
all_positions += [barycenter]
labels = n_rand * ["Permutation"] + n_rand * ["Doubly stochastic"] + ["Barycenter"]

all_positions = np.array(all_positions)

all_positions = all_positions.reshape((len(all_positions), -1))

cmds = ClassicalMDS(n_components=2, dissimilarity="euclidean")
X = cmds.fit_transform(all_positions)

plot_df = pd.DataFrame(data=X)
plot_df["label"] = labels
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=plot_df, x=0, y=1, ax=ax, hue="label")


# %% [markdown]
# ##


# # %% [markdown]
# # ## Create the matching matrix


# def diag_indices(length, k=0):
#     return (np.arange(length - k), np.arange(k, length))


# def make_flat_match(length, **kws):
#     match_mat = np.zeros((length, length))
#     match_mat[np.triu_indices(length, k=1)] = 1
#     return match_mat


# def make_linear_match(length, offset=0, **kws):
#     match_mat = np.zeros((length, length))
#     for k in np.arange(1, length):
#         match_mat[diag_indices(length, k)] = length - k + offset
#     return match_mat


# def normalize_match(graph, match_mat):
#     return match_mat / match_mat.sum() * graph.sum()


# # %% [markdown]
# # ##

# methods = [make_flat_match, make_linear_match, make_exp_match]
# names = ["Flat", "Linear", "Exp"]

# gm = GraphMatch(
#     n_init=25, init_method="rand", max_iter=80, eps=0.05, shuffle_input=True
# )
# alpha = 0.005
# match_mats = []
# permutations = []
# for method, name in zip(methods, names):
#     print(name)
#     match_mat = method(len(adj), alpha=alpha)
#     match_mat = normalize_match(adj, match_mat)
#     match_mats.append(match_mat)
#     gm.fit(match_mat, adj)
#     permutations.append(gm.perm_inds_)

# # %% [markdown]
# # ##
# from src.hierarchy import signal_flow
# from src.visualization import remove_axis
# import pandas as pd

# n_verts = len(adj)
# sf = signal_flow(adj)
# sf_perm = np.argsort(-sf)
# inds = np.arange(n_verts)

# plot_df = pd.DataFrame()
# plot_df["labels"] = labels
# plot_df["x"] = inds


# def format_order_ax(ax):
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_ylabel("")
#     ax.set_xlabel("True order")
#     ax.axis("square")


# if n_blocks > 10:
#     pal = "tab20"
# else:
#     pal = "tab10"
# color_dict = dict(zip(np.unique(labels), sns.color_palette(pal, n_colors=n_blocks)))


# def plot_diag_boxes(ax):
#     for i in range(n_blocks):
#         low = i * n_per_block - 0.5
#         high = (i + 1) * n_per_block + 0.5
#         xs = [low, high, high, low, low]
#         ys = [low, low, high, high, low]
#         ax.plot(xs, ys, color=color_dict[i], linestyle="--", linewidth=0.7, alpha=0.7)


# def calc_accuracy(block_preds):
#     acc = (block_preds == labels).astype(float).mean()
#     return acc


# def calc_abs_dist(block_preds):
#     mae = np.abs(block_preds - labels).mean()
#     return mae


# def calc_euc_dist(block_preds):
#     sse = np.sqrt(((block_preds - labels) ** 2).sum())
#     mse = sse / len(block_preds)
#     return mse


# def plot_scores(perm, ax):
#     block_preds = perm // n_per_block
#     acc = calc_accuracy(block_preds)
#     mae = calc_abs_dist(block_preds)
#     mse = calc_euc_dist(block_preds)
#     ax.text(
#         0.75,
#         0.07,
#         f"Acc. {acc:.2f}\nMAE {mae:.2f}\nMSE {mse:.2f}",
#         transform=ax.transAxes,
#     )


# # model
# fig, axs = plt.subplots(3, 6, figsize=(30, 15))

# scatter_kws = dict(
#     x="x",
#     y="y",
#     hue="labels",
#     s=7,
#     linewidth=0,
#     palette=color_dict,
#     legend=False,
#     alpha=1,
# )
# first = 0
# ax = axs[0, first]
# ax.set_title("Model (truth)")
# sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=ax, square=True)
# show_annot_array = np.zeros_like(block_probs, dtype=bool)
# show_annot_array[0, :3] = 1
# for text, show_annot in zip(
#     ax.texts, (element for row in show_annot_array for element in row)
# ):
#     text.set_visible(show_annot)
# ax.set_xticks([])
# ax.set_yticks([])

# adjplot(adj, colors=labels, ax=axs[1, first], cbar=False)
# plot_df["y"] = inds
# ax = axs[2, first]
# sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
# format_order_ax(ax)
# ax.set_ylabel("Predicted order")
# plot_diag_boxes(ax)
# plot_scores(inds, ax)

# # random
# first = 1
# remove_axis(axs[0, first])
# axs[0, first].set_title("Random")
# perm = inds.copy()
# np.random.shuffle(perm)
# adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], ax=axs[1, first], cbar=False)
# plot_df["y"] = perm
# ax = axs[2, first]
# sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
# format_order_ax(ax)
# plot_diag_boxes(ax)
# plot_scores(perm, ax)

# # signal flow
# first = 2
# remove_axis(axs[0, first])
# axs[0, first].set_title("Signal flow")
# adjplot(
#     adj[np.ix_(sf_perm, sf_perm)], colors=labels[sf_perm], ax=axs[1, first], cbar=False
# )
# plot_df["y"] = sf_perm
# ax = axs[2, first]
# sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
# format_order_ax(ax)
# plot_diag_boxes(ax)
# plot_scores(sf_perm, ax)


# # graph matching
# first = 3
# for i, (match, perm) in enumerate(zip(match_mats, permutations)):
#     axs[0, i + first].set_title(names[i])
#     # matching matrix
#     adjplot(match, ax=axs[0, i + first], cbar=False)
#     # adjacency
#     adjplot(
#         adj[np.ix_(perm, perm)], colors=labels[perm], ax=axs[1, i + first], cbar=False
#     )
#     # ranks
#     plot_df["y"] = perm
#     ax = axs[2, i + first]
#     sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
#     format_order_ax(ax)
#     plot_diag_boxes(ax)
#     plot_scores(perm, ax)


# plt.tight_layout()
# stashfig("sbm-ordering" + basename)


# # axs[0, first].set_title("Signal flow")
# # axs[0, first].set_ylabel("Match matrix")
# # axs[1, first].set_ylabel("Sorted adjacency")


# #%%


# # perm = fit_gm_exp(adj, 0.005, n_init=10)
# # perm_adj = adj[np.ix_(perm, perm)]

# # ps = calc_p_by_k(ks, perm_adj)
# # exps = exp_func(ks, alpha)


# from scipy.optimize import curve_fit


# # param_guess, _ = curve_fit(exp_func, ks, ps, p0=(alpha, 1))
# # alpha_guess = param_guess[0]
# # beta_guess = param_guess[1]
# # plt.figure()
# # sns.lineplot(x=ks, y=ps)
# # sns.lineplot(x=ks, y=exps)
# # sns.lineplot(x=ks, y=exp_func(ks, alpha_guess, beta_guess))

# # %% [markdown]
# # ##

# length = len(adj)

# ks = np.arange(1, length)


# def exp_func(k, alpha, beta=1, c=0):
#     return beta * np.exp(-alpha * (k - 1)) + c


# def make_exp_match(length, alpha=0.5, beta=1, c=0, **kws):
#     match_mat = np.zeros((length, length))
#     for k in np.arange(1, length):
#         match_mat[diag_indices(length, k)] = exp_func(k, alpha, beta, c)
#     return match_mat


# def fit_gm_exp(adj, alpha, beta=1, c=0, n_init=5, norm=False):
#     gm = GraphMatch(
#         n_init=n_init, init_method="rand", max_iter=80, eps=0.05, shuffle_input=True
#     )
#     length = len(adj)
#     match_mat = make_exp_match(length, alpha=alpha)
#     if norm:
#         match_mat = normalize_match(adj, match_mat)
#     match_mats.append(match_mat)
#     gm.fit(match_mat, adj)
#     return gm.perm_inds_


# def calc_p_by_k(ks, perm_adj):
#     length = len(perm_adj)
#     ps = []
#     for k in ks:
#         p = perm_adj[diag_indices(length, k)].mean()
#         ps.append(p)
#     return np.array(ps)


# def get_vals_by_k(ks, perm_adj):
#     ys = []
#     xs = []
#     for k in ks:
#         y = perm_adj[diag_indices(len(perm_adj), k)]
#         ys.append(y)
#         x = np.full(len(y), k)
#         xs.append(x)
#     return np.concatenate(ys), np.concatenate(xs)


# #%%
# alpha_guess = 0.005
# beta_guess = 1
# c_guess = np.mean(adj)
# opt_beta = True
# opt_c = True
# n_iter = 3
# for i in range(n_iter):
#     print(i)
#     perm = fit_gm_exp(adj, alpha_guess, beta_guess, c_guess, n_init=10)
#     perm_adj = adj[np.ix_(perm, perm)]
#     ys, xs = get_vals_by_k(ks, perm_adj)
#     ps = calc_p_by_k(ks, perm_adj)
#     exps = exp_func(ks, alpha_guess, beta_guess)
#     if opt_beta:
#         param_guess, _ = curve_fit(
#             exp_func, xs, ys, p0=(alpha_guess, beta_guess, c_guess)
#         )
#         beta_guess = 1  # param_guess[1]
#         c_guess = param_guess[2]
#     else:
#         param_guess, _ = curve_fit(exp_func, ks, ps, p0=(alpha))
#     alpha_guess = param_guess[0]

#     plt.figure()
#     sns.lineplot(x=ks, y=ps)
#     sns.lineplot(x=ks, y=exps)
#     sns.lineplot(x=ks, y=exp_func(ks, alpha_guess, beta_guess, c_guess))
#     adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)
#     plt.show()

# # %% [markdown]
# # ##
# ps = calc_p_by_k(ks, adj)
# ys, xs = get_vals_by_k(ks, adj)
# param_guess, _ = curve_fit(exp_func, xs, ys, p0=(0.05, 0.5, np.mean(adj)))
# exps = exp_func(ks, *param_guess)
# plt.figure()
# sns.lineplot(x=ks, y=ps)
# sns.lineplot(x=ks, y=exps)
# perm = fit_gm_exp(adj, param_guess[0], param_guess[1], param_guess[2], n_init=10)
# adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)

# # %% [markdown]
# # ##


# permutations = []
# alphas = [0.001, 0.005, 0.01, 0.05]
# for alpha in alphas:
#     perm = fit_gm_exp(adj, alpha, n_init=5)
#     permutations.append(perm)
#     _, _, top, _ = adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)
#     top.set_title(alpha)

# # %% [markdown]
# # ##
# fig, axs = plt.subplots(3, 4, figsize=(20, 15))

# for i, (match, perm) in enumerate(zip(match_mats, permutations)):
#     perm_adj = adj[np.ix_(perm, perm)]
#     alpha = alphas[i]

#     exp = exp_func(ks, alpha=alpha)

#     ax = axs[0, i]
#     ax.set_title(alpha)

#     sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax)
#     sns.lineplot(x=ks, y=exp, ax=ax)

#     # matching matrix
#     # adjplot(match, ax=axs[0, i], cbar=False)
#     # adjacency
#     _, _, top, _ = adjplot(perm_adj, colors=labels[perm], ax=axs[1, i], cbar=False)
#     # top.set_title(alpha)
#     # ranks
#     plot_df["y"] = perm
#     ax = axs[2, i]
#     sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
#     format_order_ax(ax)
#     plot_diag_boxes(ax)
#     plot_scores(perm, ax)
#     if i == 0:
#         ax.set_ylabel("Predicted order")

# stashfig("alpha-matters" + basename)


# # %% [markdown]
# # ## Oracles
# # permutations = []
# # alphas = [0.001, 0.005, 0.01, 0.05]
# # for alpha in alphas:
# #     perm = fit_gm_exp(adj, alpha, n_init=5)
# #     permutations.append(perm)
# #     _, _, top, _ = adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)
# #     top.set_title(alpha)

# ys, xs = get_vals_by_k(ks, adj)

# inits = [(0.05), (0.05, 1), (0.05, 1, np.mean(adj))]
# names = [r"$y = exp(-ak)$", r"$y = b \cdot exp(-ak)$", r"$y = b \cdot exp(-ak) + c$"]
# permutations = []
# params = []
# for init_params, name in zip(inits, names):
#     # just decay
#     param_guess, _ = curve_fit(exp_func, xs, ys, p0=init_params)
#     params.append(param_guess)
#     exps = exp_func(ks, *param_guess)
#     perm = fit_gm_exp(adj, *param_guess, n_init=5)
#     permutations.append(perm)

# # %% [markdown]
# # ##
# fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# for i, (name, perm) in enumerate(zip(names, permutations)):
#     perm_adj = adj[np.ix_(perm, perm)]
#     p = params[i]

#     exp = exp_func(ks, *p)

#     ax = axs[0, i]
#     ax.set_title(name)

#     sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax, label=r"$\hat{P}$")
#     sns.lineplot(x=ks, y=exp, ax=ax, label=r"Match matrix")
#     if i > 0:
#         ax.get_legend().remove()
#     ax.set_xlabel("k")
#     ax.set_xticks([])

#     # matching matrix
#     # adjplot(match, ax=axs[0, i], cbar=False)
#     # adjacency
#     _, _, top, _ = adjplot(perm_adj, colors=labels[perm], ax=axs[1, i], cbar=False)
#     # top.set_title(alpha)
#     # ranks
#     plot_df["y"] = perm
#     ax = axs[2, i]
#     sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
#     format_order_ax(ax)
#     plot_diag_boxes(ax)
#     plot_scores(perm, ax)

# axs[-1, 0].set_ylabel("Predicted order")
# axs[0, 0].set_ylabel("y")
# fig.suptitle("Oracle parameter estimates (a, b, c)", y=0.95)
# stashfig("oracle-fits" + basename)

# # %% [markdown]
# # ## EM way

# n_init = 5
# alpha_guess = 0.001
# beta_guess = 1
# c_guess = np.mean(adj)
# param_guess = np.array([alpha_guess, beta_guess, c_guess])
# n_iter = 4
# parameters = [param_guess]
# permutations = []
# for i in range(n_iter):
#     perm = fit_gm_exp(adj, *param_guess, n_init=n_init)
#     perm_adj = adj[np.ix_(perm, perm)]
#     ys, xs = get_vals_by_k(ks, perm_adj)
#     ps = calc_p_by_k(ks, perm_adj)
#     param_guess, _ = curve_fit(exp_func, xs, ys, p0=param_guess)
#     parameters.append(param_guess)
#     permutations.append(perm)

# #%%
# fig, axs = plt.subplots(3, n_iter, figsize=(5 * n_iter, 15))

# for i, (params, perm) in enumerate(zip(parameters, permutations)):
#     perm_adj = adj[np.ix_(perm, perm)]

#     exp = exp_func(ks, *params)

#     ax = axs[0, i]
#     ax.set_title(f"Iteration {i}")

#     sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax, label=r"$\hat{P}$")
#     sns.lineplot(x=ks, y=exp, ax=ax, label=r"Match matrix")
#     if i > 0:
#         ax.get_legend().remove()
#     ax.set_xlabel("k")
#     ax.set_xticks([])

#     # matching matrix
#     # adjplot(match, ax=axs[0, i], cbar=False)
#     # adjacency
#     _, _, top, _ = adjplot(perm_adj, colors=labels[perm], ax=axs[1, i], cbar=False)
#     # top.set_title(alpha)
#     # ranks
#     plot_df["y"] = perm
#     ax = axs[2, i]
#     sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
#     format_order_ax(ax)
#     plot_diag_boxes(ax)
#     plot_scores(perm, ax)

# axs[-1, 0].set_ylabel("Predicted order")
# axs[0, 0].set_ylabel("y")
# # fig.suptitle("Oracle parameter estimates (a, b, c)", y=0.95)
# stashfig("em-fits" + basename)


# # %%
