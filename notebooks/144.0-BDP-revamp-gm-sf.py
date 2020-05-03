# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from joblib import Parallel, delayed
import pandas as pd

from graspy.match import GraphMatch
from graspy.plot import heatmap
from src.cluster import get_paired_inds  # TODO fix the location of this func
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


adj = mg.adj
adj = adj[np.ix_(left_inds, left_inds)]
meta = mg.meta.iloc[left_inds].copy
meta["inds"] = range(len(meta))
print(len(meta))

# subsample_inds = np.random.choice(len(meta), 100, replace=False)
# adj = adj[np.ix_(subsample_inds, subsample_inds)]
# meta = meta.iloc[subsample_inds]

# %% [markdown]
# ## Create the matching matrix

n_verts = len(adj)


def diag_indices(length, k=0):
    return (np.arange(length - k), np.arange(k, length))


def make_flat_match(length, **kws):
    match_mat = np.zeros((length, length))
    match_mat[np.triu_indices(length, k=1)] = 1
    return match_mat


def make_linear_match(length, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = length - k + offset
    return match_mat


def make_exp_match(length, alpha=0.5, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = np.exp(-alpha * (k - 1)) + offset
    return match_mat


def normalize_match(graph, match_mat):
    return match_mat / np.linalg.norm(match_mat) * np.linalg.norm(graph)


# %% [markdown]
# ##


# ks = np.arange(1, n_verts)
# for alpha in np.geomspace(0.0005, 0.05, 10):
#     ys = np.exp(-alpha * (ks - 1))
#     sns.lineplot(x=ks, y=ys, label=f"{alpha:0.3f}", legend=False)

# %% [markdown]
# ##


alphas = np.geomspace(0.0005, 0.05, 4)

n_init = 100
basename = f"-n_init={n_init}-left-only"


perm_df = pd.DataFrame()
for alpha in alphas:
    print(alpha)
    print()
    alpha = np.round(alpha, decimals=5)
    match_mat = make_exp_match(n_verts, alpha=alpha)
    match_mat = normalize_match(adj, match_mat)

    seeds = np.random.choice(int(1e8), n_init, replace=False)

    def run_gmp(seed):
        np.random.seed(seed)
        sgm = GraphMatch(n_init=1, init_method="rand", max_iter=100, eps=0.05)
        sgm.fit(match_mat, adj)
        return sgm.score_, sgm.perm_inds_

    outs = Parallel(n_jobs=-1)(delayed(run_gmp)(seed) for seed in seeds)

    outs = list(zip(*outs))
    scores = outs[0]
    perms = outs[1]
    max_ind = np.argmax(scores)
    optimal_perm = perms[max_ind]
    perm_df[f"a{alpha}"] = optimal_perm
    perm_inds = optimal_perm
    perm_adj = adj[np.ix_(perm_inds, perm_inds)]
    perm_meta = meta.iloc[perm_inds, :].copy()

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    adjplot(
        perm_adj,
        meta=perm_meta,
        plot_type="scattermap",
        sizes=(1, 10),
        ax=ax,
        colors="merge_class",
        palette=CLASS_COLOR_DICT,
    )
    stashfig(f"adj-perm-left-alpha={alpha:.5f}")

stashcsv(perm_df, "permuatations" + basename)
stashcsv(meta, "meta" + basename)
adj_df = pd.DataFrame(adj, index=meta.index, columns=meta.index)
stashcsv(adj_df, "adj" + basename)
