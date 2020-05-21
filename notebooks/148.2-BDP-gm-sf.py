# %% [markdown]
# ##
import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from graspy.match import GraphMatch
from graspy.plot import heatmap
from src.utils import get_paired_inds
from src.data import load_metagraph
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
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


def normalize_match(graph, match_mat, method="fro"):
    if method == "fro":
        match_mat = match_mat / np.linalg.norm(match_mat) * np.linalg.norm(graph)
    elif method == "sum":
        match_mat = match_mat / np.sum(match_mat) * np.sum(graph)
    elif method is None or method is False:
        pass
    else:
        raise ValueError("invalid method")
    return match_mat


def make_exp_match(adj, alpha=0.5, beta=1, c=0, norm=False, **kws):
    length = len(adj)
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = exp_func(k, alpha, beta, c)
    match_mat = normalize_match(adj, match_mat, method=norm)
    return match_mat


def fit_gm_exp(
    adj,
    alpha,
    beta=1,
    c=0,
    n_init=5,
    norm=False,
    max_iter=50,
    eps=0.05,
    n_jobs=1,
    verbose=0,
):
    warnings.filterwarnings("ignore")
    gm = GraphMatch(
        n_init=1, init_method="rand", max_iter=max_iter, eps=eps, shuffle_input=True
    )
    match_mat = make_exp_match(adj, alpha=alpha, beta=beta, c=c, norm=norm)

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
master_mg = load_metagraph(graph_type)
mg = master_mg.remove_pdiff()
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

idx = mg.meta[mg.meta["pair"].isin(mg.meta.index)].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)

meta = mg.meta
meta["pair_td"] = meta["pair_id"].map(meta.groupby("pair_id")["Total degree"].mean())
mg = mg.sort_values(["pair_td", "pair_id"], ascending=False)
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

n_subsample = n_pairs

subsample_inds = np.random.choice(n_pairs, n_subsample, replace=False)

left_adj = left_adj[np.ix_(subsample_inds, subsample_inds)]
left_meta = left_meta.iloc[subsample_inds]

right_adj = right_adj[np.ix_(subsample_inds, subsample_inds)]
right_meta = right_meta.iloc[subsample_inds]

# %% [markdown]
# ##
pal = sns.color_palette("deep", n_colors=8)
left_color = pal[0]
right_color = pal[1]
match_color = pal[2]


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
        color=left_color,
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
        color=right_color,
    )
    if titles:
        top.set_title(r"Right $\to$ right")
    return axs


def rank_corr_plot(left_sort, right_sort, ax=None, show_corr=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(
        x=left_sort, y=right_sort, ax=ax, s=15, linewidth=0, alpha=0.8, color=pal[4]
    )
    if show_corr:
        corr = np.corrcoef(left_sort, right_sort)[0, 1]
        ax.text(
            0.75, 0.05, f"Corr. = {corr:.2f}", transform=ax.transAxes, color="black"
        )
    ax.set_xlabel("Left rank", color=left_color)
    ax.set_ylabel("Right rank", color=right_color)
    ax.set_xticks([])
    ax.set_yticks([])
    return corr


def plot_diag_vals(adj, ax, color="steelblue", kde=True, **kws):
    ks = np.arange(-len(adj) + 1, len(adj))
    vals = calc_mean_by_k(ks, adj)
    sns.scatterplot(
        x=ks, y=vals, s=10, alpha=0.4, linewidth=0, ax=ax, color=color, **kws
    )
    if kde:
        kde_vals = gaussian_filter1d(vals, sigma=25)
        sns.lineplot(x=ks, y=kde_vals, ax=ax, color=color)
    ax.set_xlabel("Diagonal index")
    line_kws = dict(linewidth=1, linestyle="--", color="grey")
    ax.axvline(0, **line_kws)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))


# %% [markdown]
# ##


n_init = 12 * 4
n_jobs = -2

currtime = time.time()

n_verts = len(left_adj)

halfs = [0.5, 1, 5, 10, 50, 100]
# halfs = [5, 10]

alphas = [np.round(np.log(2) / (h * n_verts), decimals=7) for h in halfs]
print(alphas)

param_grid = {
    "alpha": alphas,
    "beta": [1, 0.9, 0.7, 0.5, 0.3, 0.1],
    "norm": [False, "fro", "sum"],
    "c": [0],
}
params = list(ParameterGrid(param_grid))
basename = f"-n_subsample={n_subsample}"


def get_basename(n_subsample=None, alpha=None, beta=None, c=None, norm=None):
    return f"-n_subsample={n_subsample}-alpha={alpha}-beta={beta}-c={c}-norm={norm}"


def set_legend_alpha(leg, alpha=1):
    for l in leg.legendHandles:
        l.set_alpha(alpha)


# %% [markdown]
# ##


rows = []
perm_df = []
for p in tqdm(params):
    row = p.copy()
    left_row = p.copy()
    left_row["train_side"] = "left"
    right_row = p.copy()
    right_row["train_side"] = "right"
    basename = get_basename(n_subsample=n_subsample, **p)

    left_perms, left_scores = fit_gm_exp(
        left_adj, n_init=n_init, n_jobs=n_jobs, verbose=0, **p
    )
    right_perms, right_scores = fit_gm_exp(
        right_adj, n_init=n_init, n_jobs=n_jobs, verbose=0, **p
    )
    gm_left_perm, gm_left_score = get_best_run(left_perms, left_scores)
    gm_right_perm, gm_right_score = get_best_run(right_perms, right_scores)

    left_perm_series = pd.Series(data=gm_left_perm, name=str(left_row))
    right_perm_series = pd.Series(data=gm_right_perm, name=str(right_row))
    perm_df.append(left_perm_series)
    perm_df.append(right_perm_series)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    double_adj_plot(gm_left_perm, gm_right_perm, axs=axs[0, :])

    gm_left_sort = np.argsort(gm_left_perm)
    gm_right_sort = np.argsort(gm_right_perm)
    ax = axs[1, 0]
    corr = rank_corr_plot(gm_left_sort, gm_right_sort, ax=ax)
    row["corr"] = corr

    ax = axs[1, 1]
    left_perm_adj = left_adj[np.ix_(gm_left_perm, gm_left_perm)]
    plot_diag_vals(left_perm_adj, ax, color=left_color, label="Left")
    right_perm_adj = right_adj[np.ix_(gm_right_perm, gm_right_perm)]
    plot_diag_vals(right_perm_adj, ax, color=right_color, label="Right")
    match = make_exp_match(left_perm_adj, **p)
    plot_diag_vals(match, ax, color=match_color, kde=False, label="Match")
    leg = ax.legend(bbox_to_anchor=(0, 1), loc="upper left", markerscale=3)
    set_legend_alpha(leg)

    fig.suptitle(p, y=0.95)
    stashfig(f"match-profile-{p}" + basename)

    row["score"] = gm_left_score
    row["norm_score"] = gm_left_score / np.linalg.norm(match)
    row["match_fro"] = np.linalg.norm(match)
    rows.append(row)

time_mins = (time.time() - currtime) / 60
print(f"{time_mins:.2f} minutes elapsed")


res_df = pd.DataFrame(rows)
stashcsv(res_df, "res_df" + basename)

perm_df = pd.DataFrame(perm_df)
stashcsv(perm_df, "perm_df" + basename)

heatmap_kws = dict(annot=True, annot_kws={"size": 8}, cmap="Reds", vmin=0, vmax=1)
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
