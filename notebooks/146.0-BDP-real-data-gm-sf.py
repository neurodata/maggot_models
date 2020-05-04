# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from scipy.optimize import curve_fit
from graspy.match import GraphMatch
from graspy.plot import heatmap
from graspy.simulations import sbm
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


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


#%%

import pandas as pd
from src.io import readcsv


alphas = np.geomspace(0.0005, 0.05, 20)

n_init = 100
basename = f"-n_init={n_init}-left-only"

exp_name = "144.0-BDP-revamp-gm-sf"
perm_df = readcsv("permuatations" + basename, foldername=exp_name, index_col=0)
meta = readcsv("meta" + basename, foldername=exp_name, index_col=0)
# adj_df = pd.DataFrame(adj, index=meta.index, columns=meta.index)
adj_df = readcsv("adj" + basename, foldername=exp_name, index_col=0)
adj = adj_df.values
alpha = 0.00021
alpha = np.round(alpha, decimals=5)
str_alpha = f"a{alpha}"
perm_inds = perm_df[str_alpha]

perm_adj = adj[np.ix_(perm_inds, perm_inds)]
perm_meta = meta.iloc[perm_inds].copy()
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    perm_adj,
    meta=perm_meta,
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    plot_type="scattermap",
    sizes=(1, 10),
    ax=ax,
)
# %% [markdown]
# ##

from src.hierarchy import signal_flow
from src.visualization import remove_axis
import pandas as pd

n_verts = len(adj)
sf = signal_flow(adj)
sf_perm = np.argsort(-sf)
inds = np.arange(n_verts)

plot_df = pd.DataFrame()
# plot_df["labels"] = labels
plot_df["x"] = inds


def format_order_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("True order")
    ax.axis("square")


# if n_blocks > 10:
#     pal = "tab20"
# else:
#     pal = "tab10"
# color_dict = dict(zip(np.unique(labels), sns.color_palette(pal, n_colors=n_blocks)))


def plot_diag_boxes(ax):
    for i in range(n_blocks):
        low = i * n_per_block - 0.5
        high = (i + 1) * n_per_block + 0.5
        xs = [low, high, high, low, low]
        ys = [low, low, high, high, low]
        ax.plot(xs, ys, color=color_dict[i], linestyle="--", linewidth=0.7, alpha=0.7)


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


def plot_scores(perm, ax):
    block_preds = perm // n_per_block
    acc = calc_accuracy(block_preds)
    mae = calc_abs_dist(block_preds)
    mse = calc_euc_dist(block_preds)
    ax.text(
        0.75,
        0.07,
        f"Acc. {acc:.2f}\nMAE {mae:.2f}\nMSE {mse:.2f}",
        transform=ax.transAxes,
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


def exp_func(k, alpha, beta=1, c=0):
    return beta * np.exp(-alpha * (k - 1)) + c


def calc_p_by_k(ks, perm_adj):
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


def make_flat_match(length, **kws):
    match_mat = np.zeros((length, length))
    match_mat[np.triu_indices(length, k=1)] = 1
    return match_mat


def make_linear_match(length, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = length - k + offset
    return match_mat


def normalize_match(graph, match_mat):
    return match_mat / match_mat.sum() * graph.sum()


def make_exp_match(length, alpha=0.5, beta=1, c=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = exp_func(k, alpha, beta, c)
    return match_mat


def fit_gm_exp(adj, alpha, beta=1, c=0, n_init=5, norm=False):
    gm = GraphMatch(
        n_init=n_init, init_method="rand", max_iter=80, eps=0.05, shuffle_input=True
    )
    length = len(adj)
    match_mat = make_exp_match(length, alpha=alpha)
    if norm:
        match_mat = normalize_match(adj, match_mat)
    gm.fit(match_mat, adj)
    return gm.perm_inds_


# %% [markdown]
# ##

n_verts = len(adj)
ks = np.arange(1, n_verts)
n_init = 5
alpha_guess = 0.0001
beta_guess = 1
c_guess = np.mean(adj)
param_guess = np.array([alpha_guess, beta_guess, c_guess])
n_iter = 5
parameters = [param_guess]
permutations = []
for i in range(n_iter):
    perm = fit_gm_exp(adj, *param_guess, n_init=n_init)
    perm_adj = adj[np.ix_(perm, perm)]
    ys, xs = get_vals_by_k(ks, perm_adj)
    ps = calc_p_by_k(ks, perm_adj)
    param_guess, _ = curve_fit(exp_func, xs, ys, p0=param_guess)
    parameters.append(param_guess)
    permutations.append(perm)

# %% [markdown]
# ## ra
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

fig, axs = plt.subplots(2, n_iter, figsize=(5 * n_iter, 10))

for i, (params, perm) in enumerate(zip(parameters, permutations)):
    perm_adj = adj[np.ix_(perm, perm)]

    exp = exp_func(ks, *params)

    ax = axs[0, i]
    ax.set_title(f"Iteration {i}")

    sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax, label=r"$\hat{P}$")
    sns.lineplot(x=ks, y=exp, ax=ax, label=r"Match matrix")
    if i > 0:
        ax.get_legend().remove()
    ax.set_xlabel("k")
    ax.set_xticks([])

    # matching matrix
    # adjplot(match, ax=axs[0, i], cbar=False)
    # adjacency
    perm_meta = meta.iloc[perm]
    _, _, top, _ = adjplot(
        perm_adj,
        meta=perm_meta,
        colors="merge_class",
        ax=axs[1, i],
        palette=CLASS_COLOR_DICT,
        plot_type="scattermap",
        sizes=(1, 2),
    )
    # top.set_title(alpha)
    # ranks
    # plot_df["y"] = perm
    # ax = axs[2, i]
    # sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
    # format_order_ax(ax)
    # plot_diag_boxes(ax)
    # plot_scores(perm, ax)

axs[-1, 0].set_ylabel("Predicted order")
axs[0, 0].set_ylabel("y")
# fig.suptitle("Oracle parameter estimates (a, b, c)", y=0.95)
stashfig("real-em-fits" + basename)


# %% [markdown]
# ##

n_verts = len(adj)
ks = np.arange(-n_verts + 2, n_verts)
ps = calc_p_by_k(ks, perm_adj)
ys, xs = get_vals_by_k(ks, perm_adj)
exps = exp_func(ks, alpha)
plt.figure()
sns.lineplot(x=ks, y=ps)
sns.lineplot(x=ks, y=exps)

# %% [markdown]
# ##
A = np.zeros((5, 5))
A[diag_indices(5, 2)[::-1]] = 2

# %% [markdown]
# ## generate SBM
low_p = 0.01
diag_p = 0.1
feedforward_p = 0.3
n_blocks = 10
n_per_block = 50  # 50
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

methods = [make_flat_match, make_linear_match, make_exp_match]
names = ["Flat", "Linear", "Exp"]

gm = GraphMatch(
    n_init=25, init_method="rand", max_iter=80, eps=0.05, shuffle_input=True
)
alpha = 0.005
match_mats = []
permutations = []
for method, name in zip(methods, names):
    print(name)
    match_mat = method(len(adj), alpha=alpha)
    match_mat = normalize_match(adj, match_mat)
    match_mats.append(match_mat)
    gm.fit(match_mat, adj)
    permutations.append(gm.perm_inds_)

# %% [markdown]
# ##
from src.hierarchy import signal_flow
from src.visualization import remove_axis
import pandas as pd

n_verts = len(adj)
sf = signal_flow(adj)
sf_perm = np.argsort(-sf)
inds = np.arange(n_verts)

plot_df = pd.DataFrame()
plot_df["labels"] = labels
plot_df["x"] = inds


def format_order_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("True order")
    ax.axis("square")


if n_blocks > 10:
    pal = "tab20"
else:
    pal = "tab10"
color_dict = dict(zip(np.unique(labels), sns.color_palette(pal, n_colors=n_blocks)))


def plot_diag_boxes(ax):
    for i in range(n_blocks):
        low = i * n_per_block - 0.5
        high = (i + 1) * n_per_block + 0.5
        xs = [low, high, high, low, low]
        ys = [low, low, high, high, low]
        ax.plot(xs, ys, color=color_dict[i], linestyle="--", linewidth=0.7, alpha=0.7)


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


def plot_scores(perm, ax):
    block_preds = perm // n_per_block
    acc = calc_accuracy(block_preds)
    mae = calc_abs_dist(block_preds)
    mse = calc_euc_dist(block_preds)
    ax.text(
        0.75,
        0.07,
        f"Acc. {acc:.2f}\nMAE {mae:.2f}\nMSE {mse:.2f}",
        transform=ax.transAxes,
    )


# model
fig, axs = plt.subplots(3, 6, figsize=(30, 15))

scatter_kws = dict(
    x="x",
    y="y",
    hue="labels",
    s=7,
    linewidth=0,
    palette=color_dict,
    legend=False,
    alpha=1,
)
first = 0
ax = axs[0, first]
ax.set_title("Model (truth)")
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=ax, square=True)
show_annot_array = np.zeros_like(block_probs, dtype=bool)
show_annot_array[0, :3] = 1
for text, show_annot in zip(
    ax.texts, (element for row in show_annot_array for element in row)
):
    text.set_visible(show_annot)
ax.set_xticks([])
ax.set_yticks([])

adjplot(adj, colors=labels, ax=axs[1, first], cbar=False)
plot_df["y"] = inds
ax = axs[2, first]
sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
format_order_ax(ax)
ax.set_ylabel("Predicted order")
plot_diag_boxes(ax)
plot_scores(inds, ax)

# random
first = 1
remove_axis(axs[0, first])
axs[0, first].set_title("Random")
perm = inds.copy()
np.random.shuffle(perm)
adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], ax=axs[1, first], cbar=False)
plot_df["y"] = perm
ax = axs[2, first]
sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
format_order_ax(ax)
plot_diag_boxes(ax)
plot_scores(perm, ax)

# signal flow
first = 2
remove_axis(axs[0, first])
axs[0, first].set_title("Signal flow")
adjplot(
    adj[np.ix_(sf_perm, sf_perm)], colors=labels[sf_perm], ax=axs[1, first], cbar=False
)
plot_df["y"] = sf_perm
ax = axs[2, first]
sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
format_order_ax(ax)
plot_diag_boxes(ax)
plot_scores(sf_perm, ax)


# graph matching
first = 3
for i, (match, perm) in enumerate(zip(match_mats, permutations)):
    axs[0, i + first].set_title(names[i])
    # matching matrix
    adjplot(match, ax=axs[0, i + first], cbar=False)
    # adjacency
    adjplot(
        adj[np.ix_(perm, perm)], colors=labels[perm], ax=axs[1, i + first], cbar=False
    )
    # ranks
    plot_df["y"] = perm
    ax = axs[2, i + first]
    sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
    format_order_ax(ax)
    plot_diag_boxes(ax)
    plot_scores(perm, ax)


plt.tight_layout()
stashfig("sbm-ordering" + basename)


# axs[0, first].set_title("Signal flow")
# axs[0, first].set_ylabel("Match matrix")
# axs[1, first].set_ylabel("Sorted adjacency")


#%%


# perm = fit_gm_exp(adj, 0.005, n_init=10)
# perm_adj = adj[np.ix_(perm, perm)]

# ps = calc_p_by_k(ks, perm_adj)
# exps = exp_func(ks, alpha)


from scipy.optimize import curve_fit


# param_guess, _ = curve_fit(exp_func, ks, ps, p0=(alpha, 1))
# alpha_guess = param_guess[0]
# beta_guess = param_guess[1]
# plt.figure()
# sns.lineplot(x=ks, y=ps)
# sns.lineplot(x=ks, y=exps)
# sns.lineplot(x=ks, y=exp_func(ks, alpha_guess, beta_guess))

# %% [markdown]
# ##

length = len(adj)

ks = np.arange(1, length)


def exp_func(k, alpha, beta=1, c=0):
    return beta * np.exp(-alpha * (k - 1)) + c


def make_exp_match(length, alpha=0.5, beta=1, c=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = exp_func(k, alpha, beta, c)
    return match_mat


def fit_gm_exp(adj, alpha, beta=1, c=0, n_init=5, norm=False):
    gm = GraphMatch(
        n_init=n_init, init_method="rand", max_iter=80, eps=0.05, shuffle_input=True
    )
    length = len(adj)
    match_mat = make_exp_match(length, alpha=alpha)
    if norm:
        match_mat = normalize_match(adj, match_mat)
    match_mats.append(match_mat)
    gm.fit(match_mat, adj)
    return gm.perm_inds_


def calc_p_by_k(ks, perm_adj):
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


#%%
alpha_guess = 0.005
beta_guess = 1
c_guess = np.mean(adj)
opt_beta = True
opt_c = True
n_iter = 3
for i in range(n_iter):
    print(i)
    perm = fit_gm_exp(adj, alpha_guess, beta_guess, c_guess, n_init=10)
    perm_adj = adj[np.ix_(perm, perm)]
    ys, xs = get_vals_by_k(ks, perm_adj)
    ps = calc_p_by_k(ks, perm_adj)
    exps = exp_func(ks, alpha_guess, beta_guess)
    if opt_beta:
        param_guess, _ = curve_fit(
            exp_func, xs, ys, p0=(alpha_guess, beta_guess, c_guess)
        )
        beta_guess = 1  # param_guess[1]
        c_guess = param_guess[2]
    else:
        param_guess, _ = curve_fit(exp_func, ks, ps, p0=(alpha))
    alpha_guess = param_guess[0]

    plt.figure()
    sns.lineplot(x=ks, y=ps)
    sns.lineplot(x=ks, y=exps)
    sns.lineplot(x=ks, y=exp_func(ks, alpha_guess, beta_guess, c_guess))
    adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)
    plt.show()

# %% [markdown]
# ##
ps = calc_p_by_k(ks, adj)
ys, xs = get_vals_by_k(ks, adj)
param_guess, _ = curve_fit(exp_func, xs, ys, p0=(0.05, 0.5, np.mean(adj)))
exps = exp_func(ks, *param_guess)
plt.figure()
sns.lineplot(x=ks, y=ps)
sns.lineplot(x=ks, y=exps)
perm = fit_gm_exp(adj, param_guess[0], param_guess[1], param_guess[2], n_init=10)
adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)

# %% [markdown]
# ##


permutations = []
alphas = [0.001, 0.005, 0.01, 0.05]
for alpha in alphas:
    perm = fit_gm_exp(adj, alpha, n_init=5)
    permutations.append(perm)
    _, _, top, _ = adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)
    top.set_title(alpha)

# %% [markdown]
# ##
fig, axs = plt.subplots(3, 4, figsize=(20, 15))

for i, (match, perm) in enumerate(zip(match_mats, permutations)):
    perm_adj = adj[np.ix_(perm, perm)]
    alpha = alphas[i]

    exp = exp_func(ks, alpha=alpha)

    ax = axs[0, i]
    ax.set_title(alpha)

    sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax)
    sns.lineplot(x=ks, y=exp, ax=ax)

    # matching matrix
    # adjplot(match, ax=axs[0, i], cbar=False)
    # adjacency
    _, _, top, _ = adjplot(perm_adj, colors=labels[perm], ax=axs[1, i], cbar=False)
    # top.set_title(alpha)
    # ranks
    plot_df["y"] = perm
    ax = axs[2, i]
    sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
    format_order_ax(ax)
    plot_diag_boxes(ax)
    plot_scores(perm, ax)
    if i == 0:
        ax.set_ylabel("Predicted order")

stashfig("alpha-matters" + basename)


# %% [markdown]
# ## Oracles
# permutations = []
# alphas = [0.001, 0.005, 0.01, 0.05]
# for alpha in alphas:
#     perm = fit_gm_exp(adj, alpha, n_init=5)
#     permutations.append(perm)
#     _, _, top, _ = adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], cbar=False)
#     top.set_title(alpha)

ys, xs = get_vals_by_k(ks, adj)

inits = [(0.05), (0.05, 1), (0.05, 1, np.mean(adj))]
names = [r"$y = exp(-ak)$", r"$y = b \cdot exp(-ak)$", r"$y = b \cdot exp(-ak) + c$"]
permutations = []
params = []
for init_params, name in zip(inits, names):
    # just decay
    param_guess, _ = curve_fit(exp_func, xs, ys, p0=init_params)
    params.append(param_guess)
    exps = exp_func(ks, *param_guess)
    perm = fit_gm_exp(adj, *param_guess, n_init=5)
    permutations.append(perm)

# %% [markdown]
# ##
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for i, (name, perm) in enumerate(zip(names, permutations)):
    perm_adj = adj[np.ix_(perm, perm)]
    p = params[i]

    exp = exp_func(ks, *p)

    ax = axs[0, i]
    ax.set_title(name)

    sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax, label=r"$\hat{P}$")
    sns.lineplot(x=ks, y=exp, ax=ax, label=r"Match matrix")
    if i > 0:
        ax.get_legend().remove()
    ax.set_xlabel("k")
    ax.set_xticks([])

    # matching matrix
    # adjplot(match, ax=axs[0, i], cbar=False)
    # adjacency
    _, _, top, _ = adjplot(perm_adj, colors=labels[perm], ax=axs[1, i], cbar=False)
    # top.set_title(alpha)
    # ranks
    plot_df["y"] = perm
    ax = axs[2, i]
    sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
    format_order_ax(ax)
    plot_diag_boxes(ax)
    plot_scores(perm, ax)

axs[-1, 0].set_ylabel("Predicted order")
axs[0, 0].set_ylabel("y")
fig.suptitle("Oracle parameter estimates (a, b, c)", y=0.95)
stashfig("oracle-fits" + basename)

# %% [markdown]
# ## EM way

n_init = 5
alpha_guess = 0.001
beta_guess = 1
c_guess = np.mean(adj)
param_guess = np.array([alpha_guess, beta_guess, c_guess])
n_iter = 4
parameters = [param_guess]
permutations = []
for i in range(n_iter):
    perm = fit_gm_exp(adj, *param_guess, n_init=n_init)
    perm_adj = adj[np.ix_(perm, perm)]
    ys, xs = get_vals_by_k(ks, perm_adj)
    ps = calc_p_by_k(ks, perm_adj)
    param_guess, _ = curve_fit(exp_func, xs, ys, p0=param_guess)
    parameters.append(param_guess)
    permutations.append(perm)

#%%
fig, axs = plt.subplots(3, n_iter, figsize=(5 * n_iter, 15))

for i, (params, perm) in enumerate(zip(parameters, permutations)):
    perm_adj = adj[np.ix_(perm, perm)]

    exp = exp_func(ks, *params)

    ax = axs[0, i]
    ax.set_title(f"Iteration {i}")

    sns.lineplot(x=ks, y=calc_p_by_k(ks, perm_adj), ax=ax, label=r"$\hat{P}$")
    sns.lineplot(x=ks, y=exp, ax=ax, label=r"Match matrix")
    if i > 0:
        ax.get_legend().remove()
    ax.set_xlabel("k")
    ax.set_xticks([])

    # matching matrix
    # adjplot(match, ax=axs[0, i], cbar=False)
    # adjacency
    _, _, top, _ = adjplot(perm_adj, colors=labels[perm], ax=axs[1, i], cbar=False)
    # top.set_title(alpha)
    # ranks
    plot_df["y"] = perm
    ax = axs[2, i]
    sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
    format_order_ax(ax)
    plot_diag_boxes(ax)
    plot_scores(perm, ax)

axs[-1, 0].set_ylabel("Predicted order")
axs[0, 0].set_ylabel("y")
# fig.suptitle("Oracle parameter estimates (a, b, c)", y=0.95)
stashfig("em-fits" + basename)


# %%
