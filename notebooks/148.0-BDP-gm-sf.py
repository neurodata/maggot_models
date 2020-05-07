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

# %% [markdown]
# ##
def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


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


def fit_gm_exp(
    adj,
    alpha,
    beta=1,
    c=0,
    n_init=5,
    norm=False,
    max_iter=80,
    eps=0.05,
    n_jobs=1,
    verbose=0,
):
    gm = GraphMatch(
        n_init=1, init_method="rand", max_iter=max_iter, eps=eps, shuffle_input=True
    )
    length = len(adj)
    match_mat = make_exp_match(length, alpha=alpha)
    if norm:
        match_mat = normalize_match(adj, match_mat)

    seeds = np.random.choice(int(1e8), size=n_init)

    def _fit(seed):
        np.random.seed(seed)
        gm.fit(match_mat, adj)
        return gm.perm_inds_, gm.score_

    outs = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_fit)(s) for s in seeds)
    outs = list(zip(*outs))
    perms = np.array(outs[0])
    scores = np.array(outs[1])
    return perms, scores


def get_best_run(perms, scores, n_opts=None):
    if n_opts is None:
        n_opts = len(perms)
    opt_inds = np.random.choice(len(perms), n_opts, replace=False)
    perms = perms[opt_inds]
    scores = scores[opt_inds]
    max_ind = np.argmax(scores)
    return perms[max_ind], scores[max_ind]


# %% [markdown]
# ##
np.random.seed(8888)

graph_type = "G"
master_mg = load_metagraph(graph_type, version="2020-04-23")
mg = preprocess(
    master_mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

degrees = mg.calculate_degrees()
quant_val = np.quantile(degrees["Total edgesum"], 0.05)

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > quant_val].index
print(quant_val)
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

idx = mg.meta[mg.meta["Pair"].isin(mg.meta.index)].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)

meta = mg.meta
meta["pair_td"] = meta["Pair ID"].map(meta.groupby("Pair ID")["Total degree"].mean())
mg = mg.sort_values(["pair_td", "Pair ID"], ascending=False)
meta["inds"] = range(len(meta))
adj = mg.adj.copy()
lp_inds, rp_inds = get_paired_inds(meta)
left_inds = meta[meta["left"]]["inds"]

n_pairs = len(lp_inds)

adj = mg.adj
left_adj = adj[np.ix_(lp_inds, lp_inds)]
left_meta = mg.meta.iloc[lp_inds].copy()

right_adj = adj[np.ix_(rp_inds, rp_inds)]
right_meta = mg.meta.iloc[rp_inds].copy()

# %% [markdown]
# ##
np.random.seed(8888)
# n_subsample = n_pairs // 2
n_subsample = 200

subsample_inds = np.random.choice(n_pairs, n_subsample, replace=False)

left_adj = left_adj[np.ix_(subsample_inds, subsample_inds)]
left_meta = left_meta.iloc[subsample_inds]

right_adj = right_adj[np.ix_(subsample_inds, subsample_inds)]
right_meta = right_meta.iloc[subsample_inds]

# %% [markdown]
# ##


def double_adj_plot(left_perm, right_perm, axs=None, titles=True):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    left_perm_adj = left_adj[np.ix_(left_perm, left_perm)]
    left_perm_meta = left_meta.iloc[left_perm]
    ax = axs[0]
    _, _, top, _ = adjplot(
        left_perm_adj,
        meta=left_perm_meta,
        plot_type="scattermap",
        sizes=(1, 10),
        ax=ax,
        colors="merge_class",
        palette=CLASS_COLOR_DICT,
    )
    if titles:
        top.set_title(r"Left $\to$ left")

    right_perm_adj = right_adj[np.ix_(right_perm, right_perm)]
    right_perm_meta = right_meta.iloc[right_perm]
    ax = axs[1]
    _, _, top, _ = adjplot(
        right_perm_adj,
        meta=right_perm_meta,
        plot_type="scattermap",
        sizes=(1, 10),
        ax=ax,
        colors="merge_class",
        palette=CLASS_COLOR_DICT,
    )
    if titles:
        top.set_title(r"Right $\to$ right")
    return axs


def rank_corr_plot(left_sort, right_sort, ax=None, show_corr=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(x=left_sort, y=right_sort, ax=ax, s=15, linewidth=0, alpha=0.8)
    if show_corr:
        corr = np.corrcoef(left_sort, right_sort)[0, 1]
        ax.text(0.75, 0.05, f"Corr. = {corr:.2f}", transform=ax.transAxes)
    ax.set_xlabel("Left rank")
    ax.set_ylabel("Right rank")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# %% [markdown]
# ##
alpha = 0.013  # 0.0001
beta = 0.72  # 1
c = 0
n_init = 12 * 4
norm = False
n_jobs = -2
basename = f"-n_subsample={n_subsample}-alpha={alpha}-beta={beta}-c={c}-norm={norm}"

currtime = time.time()
left_perms, left_scores = fit_gm_exp(
    left_adj,
    alpha=alpha,
    beta=beta,
    c=c,
    n_init=n_init,
    norm=norm,
    n_jobs=n_jobs,
    verbose=10,
)
right_perms, right_scores = fit_gm_exp(
    right_adj,
    alpha=alpha,
    beta=beta,
    c=c,
    n_init=n_init,
    norm=norm,
    n_jobs=n_jobs,
    verbose=10,
)

time_mins = (time.time() - currtime) / 60
print(f"{time_mins:.2f} minutes elapsed")

# %% [markdown]
# ##
perms, scores = right_perms, right_scores

rows = []
n_tries = 100
for n_opts in range(1, n_init + 1):
    for i in range(n_tries):
        best_perm, best_score = get_best_run(perms, scores, n_opts=n_opts)
        row = {"score": best_score, "n_opts": n_opts}
        rows.append(row)

score_df = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(x="n_opts", y="score", data=score_df)
stashfig("n_init" + basename)

# %% [markdown]
# ##
# need to look at the order in which each pair is given in the permutation indices
# left_perm = np.array([2, 1, 0, 3, 4])
# right_perm = np.array([2, 4, 0, 1, 3])
# these you read as "put the 0th thing in position 3"
# print(f"Items {np.sort(right_perm)} are ordered as:")
# print(np.argsort(left_perm))
# print(np.argsort(right_perm))
# this you read as "thing at posiition i goes to arr[i]" so the rank of thing at i

gm_left_perm, gm_left_score = get_best_run(left_perms, left_scores)
gm_right_perm, gm_right_score = get_best_run(right_perms, right_scores)

double_adj_plot(gm_left_perm, gm_right_perm)
stashfig("adj-gm-flow" + basename)

gm_left_sort = np.argsort(gm_left_perm)
gm_right_sort = np.argsort(gm_right_perm)
ax = rank_corr_plot(gm_left_sort, gm_right_sort)
ax.set_title("Pair graph match flow")
stashfig("rank-gm-flow" + basename)

# signal flow
left_sf = -signal_flow(left_adj)
right_sf = -signal_flow(right_adj)

# how to permute the graph to sort in signal flow
sf_left_perm = np.argsort(left_sf)
sf_right_perm = np.argsort(right_sf)
double_adj_plot(sf_left_perm, sf_right_perm)
stashfig("adj-signal-flow" + basename)

# how things get ranked in terms of the above
sf_left_sort = np.argsort(sf_left_perm)
sf_right_sort = np.argsort(sf_right_perm)
ax = rank_corr_plot(sf_left_sort, sf_right_sort)
ax.set_title("Pair signal flow")
stashfig("rank-signal-flow" + basename)

# %% [markdown]
# ## plot all together
scale = 8
n_col = 4
n_row = 2
pad = 20

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(n_col * scale, n_row * scale))
gs = GridSpec(n_row, n_col, figure=fig)
axs = np.empty((n_row, n_col - 1), dtype="O")
for i in range(n_row):
    for j in range(n_col - 1):
        axs[i, j] = fig.add_subplot(gs[i, j])

double_adj_plot(sf_left_perm, sf_right_perm, axs=axs[0, :2], titles=False)
double_adj_plot(gm_left_perm, gm_right_perm, axs=axs[1, :2], titles=False)
rank_corr_plot(sf_left_sort, sf_right_sort, ax=axs[0, 2])
rank_corr_plot(gm_left_sort, gm_right_sort, ax=axs[1, 2])
axs[0, 0].set_ylabel("Signal flow", labelpad=pad)
axs[1, 0].set_ylabel("Graph match flow", labelpad=pad)
axs[0, 0].set_title(r"Left $\to$ left", pad=pad)
axs[0, 1].set_title(r"Right $\to$ right", pad=pad)
axs[0, 2].set_title("Ranks for pairs")

ax = fig.add_subplot(gs[:, -1])
sf_all_sort = np.concatenate((sf_left_sort, sf_right_sort))
gm_all_sort = np.concatenate((gm_left_sort, gm_right_sort))
rank_corr_plot(sf_all_sort, gm_all_sort, ax=ax)
ax.set_xlabel("Signal flow rank")
ax.set_ylabel("Graph match flow rank")
ax.axis("square")
ax.set_title("Ranks by method")

plt.tight_layout()
stashfig("combined" + basename)

# %% [markdown]
# ## look at the signal flow sorting in terms of diagonals
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

ks = np.arange(-n_pairs + 1, n_pairs)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
means_by_k = calc_mean_by_k(ks, left_adj[np.ix_(sf_left_perm, sf_left_perm)])
kde_vals = gaussian_filter1d(means_by_k, sigma=25)
sns.scatterplot(x=ks, y=means_by_k, s=10, alpha=0.4, linewidth=0, ax=ax, label="Left")
sns.lineplot(x=ks, y=kde_vals, ax=ax)
means_by_k = calc_mean_by_k(ks, right_adj[np.ix_(sf_right_perm, sf_right_perm)])
kde_vals = gaussian_filter1d(means_by_k, sigma=25)
sns.scatterplot(x=ks, y=means_by_k, s=10, alpha=0.4, linewidth=0, ax=ax, label="Right")
sns.lineplot(x=ks, y=kde_vals, ax=ax)
ax.set_xlabel("Diagonal index (k)")
ax.set_ylabel("Mean")
ax.set_title("Signal flow")
stashfig("signal-flow-diagplot" + basename)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
means_by_k = calc_mean_by_k(ks, left_adj[np.ix_(gm_left_perm, gm_left_perm)])
kde_vals = gaussian_filter1d(means_by_k, sigma=25)
sns.scatterplot(x=ks, y=means_by_k, s=10, alpha=0.4, linewidth=0, ax=ax, label="Left")
sns.lineplot(x=ks, y=kde_vals, ax=ax)
means_by_k = calc_mean_by_k(ks, right_adj[np.ix_(gm_right_perm, gm_right_perm)])
kde_vals = gaussian_filter1d(means_by_k, sigma=25)
sns.scatterplot(x=ks, y=means_by_k, s=10, alpha=0.4, linewidth=0, ax=ax, label="Right")
sns.lineplot(x=ks, y=kde_vals, ax=ax)

match_mat = make_exp_match(n_pairs, alpha=alpha, beta=beta, c=c)
left_match_mat = normalize_match(left_adj, match_mat)
right_match_mat = normalize_match(right_adj, match_mat)
left_vals = calc_mean_by_k(ks, left_match_mat)
right_vals = calc_mean_by_k(ks, right_match_mat)
sns.lineplot(x=ks, y=left_vals, color="purple")
sns.lineplot(x=ks, y=right_vals, color="red")
ax.set_xlabel("Diagonal index (k)")
ax.set_ylabel("Mean")
ax.set_title("Graph match flow")
stashfig("gm-flow-diagplot" + basename)


# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

alpha_guess = alpha
beta_guess = 1
ks = np.arange(n_pairs)
ys, xs = get_vals_by_k(ks, left_adj[np.ix_(gm_left_perm, gm_left_perm)])
param_guess, _ = curve_fit(exp_func, xs, ys, p0=(alpha_guess, beta_guess))

curve_proj = exp_func(ks, *param_guess)

means_by_k = calc_mean_by_k(ks, left_adj[np.ix_(gm_left_perm, gm_left_perm)])
kde_vals = gaussian_filter1d(means_by_k, sigma=25)
sns.scatterplot(x=ks, y=means_by_k, s=10, alpha=0.4, linewidth=0, ax=ax, label="Left")
match_mat = make_exp_match(n_pairs, alpha=alpha, beta=beta, c=c)
left_match_mat = normalize_match(left_adj, match_mat)
# right_match_mat = normalize_match(right_adj, match_mat)
left_vals = calc_mean_by_k(ks, left_match_mat)
# right_vals = calc_mean_by_k(ks, right_match_mat)
sns.lineplot(x=ks, y=left_vals, color="purple")
# sns.lineplot(x=ks, y=right_vals, color="red")
sns.lineplot(x=ks, y=curve_proj, color="red")
ax.set_xlabel("Diagonal index (k)")
ax.set_ylabel("Mean")
ax.set_title("Graph match flow")

