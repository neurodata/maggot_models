#%%


import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import matched_stripplot
from numba import jit
from scipy.optimize import linear_sum_assignment
from src.data import load_maggot_graph
from src.io import savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    matrixplot,
    set_theme,
)

t0 = time.time()
set_theme()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/connectivity_pair_rank/figs",
        format="pdf",
        **kws,
    )
    savefig(
        name,
        pathname="./maggot_models/experiments/connectivity_pair_rank/figs",
        format="png",
        **kws,
    )


set_theme()

colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right", "Contra"], [colors[0], colors[1], colors[3]]))

#%% [markdown]
# ### Load the data
#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["neurons"]]  # brain neurons
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
mg.nodes["inds"] = np.arange(len(mg.nodes))
left_inds, right_inds = get_paired_inds(mg.nodes)
adj = mg.sum.adj
ll_adj = adj[np.ix_(left_inds, left_inds)]
rr_adj = adj[np.ix_(right_inds, right_inds)]
lr_adj = adj[np.ix_(left_inds, right_inds)]
rl_adj = adj[np.ix_(right_inds, left_inds)]


#%%
plot_kws = dict(plot_type="scattermap", sizes=(1, 1))
fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw=dict(hspace=0, wspace=0))

ax = axs[0, 0]
adjplot(ll_adj, ax=ax, color=palette["Left"], **plot_kws)

ax = axs[0, 1]
matrixplot(
    lr_adj,
    ax=ax,
    color=palette["Contra"],
    square=True,
    **plot_kws,
)

ax = axs[1, 0]
matrixplot(
    rl_adj,
    ax=ax,
    color=palette["Contra"],
    square=True,
    **plot_kws,
)

ax = axs[1, 1]
adjplot(rr_adj, ax=ax, color=palette["Right"], **plot_kws)
#%% [markdown]
# ## Include the contralateral connections in graph matching
#%% [markdown]
# ### Run the graph matching experiment
#%%
from graspologic.match.qap import _doubly_stochastic

np.random.seed(8888)
maxiter = 30
verbose = 1
ot = False
maximize = True
reg = np.nan  # TODO could try GOAT
thr = np.nan
tol = 1e-3
n_init = 20
alpha = 0.1
n = len(ll_adj)
# construct an initialization
P0 = 1 / n * np.ones((n, n))
# P0
# P0 =
# P0[np.arange(n_pairs), np.arange(n_pairs)] = 1
# P0[n_pairs:, n_pairs:] = 1 / (n - n_pairs)


@jit(nopython=True)
def compute_gradient(A, B, AB, BA, P):
    return A @ P @ B.T + A.T @ P @ B + AB @ P.T @ BA.T + BA.T @ P.T @ AB


@jit(nopython=True)
def compute_step_size(A, B, AB, BA, P, Q):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares
    a_cross = np.trace(AB.T @ R @ BA @ R)
    b_cross = np.trace(AB.T @ R @ BA @ Q) + np.trace(AB.T @ Q @ BA @ R)
    a_intra = np.trace(A @ R @ B.T @ R.T)
    b_intra = np.trace(A @ Q @ B.T @ R.T + A @ R @ B.T @ Q.T)

    a = a_cross + a_intra
    b = b_cross + b_intra

    if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
        alpha = -b / (2 * a)
    return alpha
    # else:
    #     alpha = np.argmin([0, (b + a) * obj_func_scalar])
    # return alpha


@jit(nopython=True)
def compute_objective_function(A, B, AB, BA, P):
    return np.trace(A @ P @ B.T @ P.T) + np.trace(AB.T @ P @ BA @ P)


rows = []
for init in range(n_init):
    if verbose > 0:
        print(f"Initialization: {init}")
    shuffle_inds = np.random.permutation(n)
    correct_perm = np.argsort(shuffle_inds)
    A_base = ll_adj.copy()
    B_base = rr_adj.copy()
    AB_base = lr_adj.copy()
    BA_base = rl_adj.copy()

    for between_term in [False]:
        init_t0 = time.time()
        if verbose > 0:
            print(f"Between term: {between_term}")
        A = A_base
        B = B_base[shuffle_inds][:, shuffle_inds]
        AB = AB_for_obj = AB_base[:, shuffle_inds]
        BA = BA_for_obj = BA_base[shuffle_inds]

        if not between_term:
            AB = np.zeros((n, n))
            BA = np.zeros((n, n))

        P = P0.copy()
        P = P[:, shuffle_inds]

        if alpha > 0:
            rand_ds = np.random.uniform(size=(n, n))
            rand_ds = _doubly_stochastic(rand_ds)
            P = (1 - alpha) * P + alpha * rand_ds

        # _, iteration_perm = linear_sum_assignment(-P)
        # match_ratio = (correct_perm == iteration_perm)[:n_pairs].mean()
        # print(match_ratio)

        obj_func_scalar = 1
        if maximize:
            obj_func_scalar = -1

        for n_iter in range(1, maxiter + 1):

            # [1] Algorithm 1 Line 3 - compute the gradient of f(P)
            currtime = time.time()
            grad_fp = compute_gradient(A, B, AB, BA, P)
            if verbose > 1:
                print(f"{time.time() - currtime:.3f} seconds elapsed for grad_fp.")

            # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
            currtime = time.time()
            if ot:
                # TODO not implemented here yet
                Q = alap(grad_fp, n, maximize, reg, thr)
            else:
                _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
                Q = np.eye(n)[cols]
            if verbose > 1:
                print(
                    f"{time.time() - currtime:.3f} seconds elapsed for LSAP/Sinkhorn step."
                )

            # [1] Algorithm 1 Line 5 - compute the step size
            currtime = time.time()

            alpha = compute_step_size(A, B, AB, BA, P, Q)

            if verbose > 1:
                print(
                    f"{time.time() - currtime:.3f} seconds elapsed for quadradic terms."
                )

            # [1] Algorithm 1 Line 6 - Update P
            P_i1 = alpha * P + (1 - alpha) * Q
            if np.linalg.norm(P - P_i1) / np.sqrt(n) < tol:
                P = P_i1
                break
            P = P_i1
            # _, iteration_perm = linear_sum_assignment(-P)
            # match_ratio = (correct_perm == iteration_perm)[:n_pairs].mean()

            objfunc = compute_objective_function(A, B, AB_for_obj, BA_for_obj, P)

            if verbose > 0:
                print(f"Iteration: {n_iter},  Objective function: {objfunc:.2f}")

            row = {
                "init": init,
                "iter": n_iter,
                "objfunc": objfunc,
                # "match_ratio": match_ratio,
                "between_term": between_term,
                "time": time.time() - init_t0,
                "P": P[:, correct_perm],
            }
            rows.append(row)

        if verbose > 0:
            print("\n")

    _, perm = linear_sum_assignment(-P)
    if verbose > 0:
        print("\n")

results = pd.DataFrame(rows)
results

#%%
last_results_idx = results.groupby(["between_term", "init"])["iter"].idxmax()
last_results = results.loc[last_results_idx].copy()

total_objfunc = last_results["objfunc"].sum()
amalgam_P = np.zeros((n, n))
for idx, row in last_results.iterrows():
    P = row["P"]
    objfunc = row["objfunc"]
    scaled_P = P * objfunc / total_objfunc
    amalgam_P += scaled_P

#%%
from scipy.stats import rankdata

pair_ranks = []
for i, row in enumerate(amalgam_P):
    ranks = rankdata(-row)
    pair_rank = ranks[i]
    if i == 0:
        print(pair_rank)
    pair_ranks.append(pair_rank)
pair_ranks = np.array(pair_ranks)

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
colors = sns.color_palette()
sns.histplot(
    x=pair_ranks, ax=ax, binwidth=1, stat="probability", discrete=True, color=colors[1]
)
ax.set(ylim=(0, 1), xlim=(0.5, 5.5))
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylabel("Fraction Neuron Pairs")
ax.set_xlabel("GM rank")
ax.spines["right"].set_visible(True)
ax.spines["top"].set_visible(True)
stashfig("GM-rank")

#%%

# I = np.eye(n_pairs)
# A = A_base[:n_pairs, :n_pairs].copy()
# B = B_base[:n_pairs, :n_pairs].copy()
# AB = AB_base[:n_pairs, :n_pairs].copy()
# BA = BA_base[:n_pairs, :n_pairs].copy()
# P = amalgam_P[:n_pairs, :n_pairs].copy()

# og_objfunc = compute_objective_function(A, B, AB, BA, I)
# print(f"Objective function with current pairs: {og_objfunc}")

# for i in range(10):
#     shuffle_inds = np.random.permutation(len(P))
#     unshuffle_inds = np.argsort(shuffle_inds)
#     P_shuffle = P[shuffle_inds][:, shuffle_inds].copy()
#     _, perm_inds = linear_sum_assignment(P_shuffle, maximize=True)
#     perm_inds = perm_inds[unshuffle_inds]
#     P_final = np.eye(n)
#     P_final = P_final[perm_inds][:n_pairs, :n_pairs].copy()  # double check

#     final_objfunc = compute_objective_function(A, B, AB, BA, P_final)
#     print(f"Final objective function with predicted pairs: {final_objfunc}")


#%%


def predict_pairs(P, source_nodes, target_nodes):
    source_nodes["predicted_pairs"] = ""
    source_nodes["prediction_changed"] = False
    source_nodes["multiple_predictions"] = False
    source_nodes["had_pair"] = True
    for i in range(n):
        idx = source_nodes.index[i]
        row = P[i]
        nonzero_inds = np.nonzero(row)[0]
        sort_inds = np.argsort(-row[nonzero_inds])
        ranking = nonzero_inds[sort_inds]
        ranking_skids = target_nodes.index[ranking]
        source_nodes.at[idx, "predicted_pairs"] = list(ranking_skids)
        current_pair = source_nodes.loc[idx, "pair"]
        if ranking_skids[0] != current_pair:
            source_nodes.loc[idx, "prediction_changed"] = True
        if len(ranking_skids) > 1:
            source_nodes.loc[idx, "multiple_predictions"] = True
        if source_nodes.loc[idx, "pair"] == -1 or idx < 1:
            source_nodes.loc[idx, "had_pair"] = False
    return source_nodes


left_nodes = predict_pairs(amalgam_P, left_nodes, right_nodes)
right_nodes = predict_pairs(amalgam_P.T, right_nodes, left_nodes)
changed_left_nodes = left_nodes[
    left_nodes["prediction_changed"] | left_nodes["multiple_predictions"]
].sort_values(
    ["had_pair", "prediction_changed", "multiple_predictions"], ascending=False
)
changed_right_nodes = right_nodes[
    right_nodes["prediction_changed"] | right_nodes["multiple_predictions"]
].sort_values(
    ["had_pair", "prediction_changed", "multiple_predictions"], ascending=False
)


changed_left_nodes.to_csv(output_path / "changed_left_pairs_meta.csv")
changed_right_nodes.to_csv(output_path / "changed_right_pairs_meta.csv")

#%%

plot_nodes = changed_left_nodes[
    changed_left_nodes["prediction_changed"] & changed_left_nodes["had_pair"]
]
skeleton_palette = dict(
    zip(mg.nodes.index, np.vectorize(CLASS_COLOR_DICT.get)(mg.nodes["merge_class"]))
)
start_instance()
n_plot = 40
for i in np.random.choice(len(plot_nodes), size=n_plot, replace=False):
    idx = plot_nodes.index[i]
    row = plot_nodes.loc[idx]
    predicted_pairs = row["predicted_pairs"]
    predicted_pairs += [idx]
    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    simple_plot_neurons(predicted_pairs, palette=skeleton_palette, ax=ax, dist=6)
    ax.set_title(f"Left: {idx}")
    ax.set_xlim((7000, 99000))
    ax.set_ylim((7000, 99000))
    stashfig(f"left-{idx}-pair-predictions")

# %% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
