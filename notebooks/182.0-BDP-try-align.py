# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ortho_group
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.inference import LatentDistributionTest
from graspologic.plot import pairplot
from graspologic.utils import pass_to_ranks, augment_diagonal

# screeplot(pass_to_ranks(left_mg.adj), cumulative=False, show_first=40)
from graspologic.embed import select_dimension

# from grasp.plot import screeplot
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.visualization import adjplot, set_theme

set_theme()

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)
    savefig(name, foldername=FNAME, save_on=True, fmt="png", dpi=300, **kws)


# %%
mg = load_metagraph("G")

pair_meta = pd.read_csv(
    "maggot_models/experiments/graph_match/outs/pair_meta.csv", index_col=0
)
pair_key = "pair_id"
pair_meta = pair_meta[pair_meta[f"{pair_key[:-3]}"].isin(pair_meta.index)]
pair_meta = pair_meta.sort_values(["hemisphere", pair_key])
mg = mg.reindex(pair_meta.index.values, use_ids=True)
mg = MetaGraph(mg.adj, pair_meta)
n_pairs = len(pair_meta) // 2
left_inds = np.arange(n_pairs)
right_inds = left_inds.copy() + n_pairs
left_mg = MetaGraph(mg.adj[np.ix_(left_inds, left_inds)], mg.meta.iloc[left_inds])
right_mg = MetaGraph(mg.adj[np.ix_(right_inds, right_inds)], mg.meta.iloc[right_inds])

assert (left_mg.meta[pair_key].values == right_mg.meta[pair_key].values).all()

print(f"Working with {n_pairs} pairs.")

palette = dict(zip(["Left", "Right", "OP", "SP"], sns.color_palette("Set1")))

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
adjplot(
    left_mg.adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[0],
    title=r"Left $\to$ left",
    color=palette["Left"],
)
adjplot(
    right_mg.adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[1],
    title=r"Right $\to$ right",
)
stashfig("left-right-adjs")

#%%


def plot_latents(left, right, title=""):
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(plot_data[:, :4], labels=labels, title=title)
    return pg


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    plt.scatter(
        elbow_inds, sing_vals[elbow_inds - 1], marker="x", s=50, zorder=10, color=color
    )
    return ax


def embed(adj, n_components=40):
    elbow_inds, elbow_vals = select_dimension(
        augment_diagonal(pass_to_ranks(adj)), n_elbows=4
    )
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(pass_to_ranks(adj))
    return out_latent, in_latent, ase.singular_values_, elbow_inds


n_components = 8
max_n_components = 40
left_out_latent, left_in_latent, left_sing_vals, left_elbow_inds = embed(
    left_mg.adj, n_components=max_n_components
)
right_out_latent, right_in_latent, right_sing_vals, right_elbow_inds = embed(
    right_mg.adj, n_components=max_n_components
)

# plot the screeplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(left_sing_vals, left_elbow_inds, color=palette["Left"], ax=ax, label="Left")
screeplot(
    right_sing_vals, right_elbow_inds, color=palette["Right"], ax=ax, label="Right"
)
ax.legend()
ax.axvline(n_components, color="black", linewidth=1.5, linestyle="--")
stashfig("screeplot")

# plot the latent positions before any kind of alignment
plot_latents(
    left_out_latent, right_out_latent, title="Out latent positions (no alignment)"
)
stashfig("out-latent-no-align")
plot_latents(
    left_in_latent, right_in_latent, title="In latent positions (no alignment)"
)
stashfig("in-latent-no-align")

# %%


def run_alignments(X, Y):
    op = OrthogonalProcrustes()
    X_trans_op = op.fit_transform(X, Y)
    sp = SeedlessProcrustes(init="custom", initial_Q=op.Q_)
    X_trans_sp = sp.fit_transform(X, Y)
    return X_trans_op, X_trans_sp


def calc_diff_norm(X, Y):
    return np.linalg.norm(X - Y, ord="fro")


op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent, right_out_latent
)
op_diff_norm = calc_diff_norm(op_left_out_latent, right_out_latent)
sp_diff_norm = calc_diff_norm(sp_left_out_latent, right_out_latent)

print(f"Procrustes diff. norm using true pairs: {op_diff_norm}")
print(f"Seedless Procrustes diff. norm using true pairs: {op_diff_norm}")
#%%
plot_latents(
    op_left_out_latent, right_out_latent, "Out latent positions (Procrustes alignment)"
)
stashfig("out-latent-procrustes")

plot_latents(
    sp_left_out_latent,
    right_out_latent,
    "Out latent positions (Seedless Procrustes alignment, oracle init)",
)
stashfig("out-latent-seedless-oracle")


# %%
op_left_in_latent, sp_left_in_latent = run_alignments(left_in_latent, right_in_latent)

# %% [markdown]
# ##


def calc_nn_ranks(target, query):
    n_pairs = len(target)
    nn = NearestNeighbors(n_neighbors=n_pairs, metric="euclidean")
    nn.fit(target)
    neigh_inds = nn.kneighbors(query, return_distance=False)
    true_neigh_inds = np.arange(n_pairs)
    _, ranks = np.where((neigh_inds == true_neigh_inds[:, None]))
    return ranks


def plot_rank_cdf(x, ax=None, color=None, label=None, max_rank=51):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    hist, _ = np.histogram(x, bins=np.arange(0, max_rank, 1))
    hist = hist / n_pairs
    ax.plot(np.arange(1, max_rank, 1), hist.cumsum(), color=color, label=label)
    ax.set(ylabel="Cumulative P by K", xlabel="K")
    return ax


op_left_composite_latent = np.concatenate(
    (op_left_out_latent, op_left_in_latent), axis=1
)
sp_left_composite_latent = np.concatenate(
    (sp_left_out_latent, sp_left_in_latent), axis=1
)
right_composite_latent = np.concatenate((right_out_latent, right_in_latent), axis=1)


op_ranks = calc_nn_ranks(op_left_composite_latent, right_composite_latent)
sp_ranks = calc_nn_ranks(sp_left_composite_latent, right_composite_latent)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_rank_cdf(op_ranks, color=palette["OP"], ax=ax, label="OP")
plot_rank_cdf(sp_ranks, color=palette["SP"], ax=ax, label="SP")
ax.legend()


# %% [markdown]
# ##


def run_alignments(X, Y, n_random=10):
    n_components = X.shape[1]
    rows = []

    for i in range(n_random):
        Q = ortho_group.rvs(n_components)
        diff_norm = calc_diff_norm(X @ Q, Y)
        row = {"diff_norm": diff_norm, "method": "random"}
        rows.append(row)

    op = OrthogonalProcrustes()
    X_trans_op = op.fit_transform(X, Y)
    diff_norm = calc_diff_norm(X_trans_op, Y)
    row = {"diff_norm": diff_norm, "method": "orthogonal_procrustes"}
    rows.append(row)

    sp = SeedlessProcrustes(init="custom", initial_Q=op.Q_)
    X_trans_sp = sp.fit_transform(X, Y)
    diff_norm = calc_diff_norm(X_trans_sp, Y)
    row = {"diff_norm": diff_norm, "method": "orthogonal_procrustes"}
    rows.append(row)


# %% [markdown]
# ##


#
# ldt = LatentDistributionTest(input_graph=False)
# ldt.fit(trans_left_out_latent, right_out_latent)
# print(ldt.p_value_)


# %% [markdown]
## parameters of the experiment
# - dimension:
# - ipsi / contra
# - ptr-pre / ptr-post / no ptr
# - direction: in / out
# - alignment: OP / sP
# - test on: all / known pairs only

#%%
max_n_components = 40
ase = AdjacencySpectralEmbed(n_components=max_n_components)
left_out_latent, left_in_latent = ase.fit_transform(pass_to_ranks(left_mg.adj))
right_out_latent, right_in_latent = ase.fit_transform(pass_to_ranks(right_mg.adj))


def align(X, Y, method="procrustes"):
    if method == "procrustes":
        op = OrthogonalProcrustes()
        X_trans = op.fit_transform(X, Y)
    return X_trans


def test(X, Y):
    ldt = LatentDistributionTest(input_graph=False)
    ldt.fit(X, Y)
    return ldt.p_value_, ldt.sample_T_statistic_


def run_align_and_test(X, Y, n_components=None):
    if n_components is not None:
        X = X[:, :n_components]
        Y = Y[:, :n_components]
    start_time = time.time()
    X = align(X, Y)
    p_value, test_statistic = test(X, Y)
    elapsed = time.time() - start_time
    return {
        "p_value": p_value,
        "test_statistic": test_statistic,
        "time": elapsed,
        "log_p_value": np.log10(p_value),
    }


#%%


rows = []
n_components_range = np.arange(1, max_n_components)
in_outs = ["in", "out"]
in_out_to_latent = {
    "in": (left_in_latent, right_in_latent),
    "out": (left_out_latent, right_out_latent),
}
n_tests = len(n_components_range) * len(in_outs)

i = 0
for in_out in in_outs:
    for n_components in n_components_range:
        print(f"{i / (n_tests - 1):.02f}")
        left_latent, right_latent = in_out_to_latent[in_out]
        row = run_align_and_test(left_latent, right_latent, n_components=n_components)
        row["n_components"] = n_components
        row["in_out"] = in_out
        rows.append(row)
        i += 1

result_df = pd.DataFrame(rows)

# %% [markdown]
# ##


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    x="n_components", y="log_p_value", hue="in_out", data=result_df, ax=ax, marker="o"
)
# sns.scatterplot(
#     x="n_components", y="log_p_value", hue="in_out", data=result_df, s=30, ax=ax
# )
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
ax.legend(
    bbox_to_anchor=(1, 1), loc="upper left", handles=handles[1:], labels=labels[1:]
)
ax.set(
    ylabel="Log10(p-value)",
    xlabel="# dimensions",
    title="Latent distribution test (known + predicted pair procrustes)",
)
stashfig("p-val-sweep")
# %% [markdown]
# ##
# plt.plot(ase.singular_values_)


# print(calc_diff_norm(op_left_out_latent, right_out_latent))
# print(calc_diff_norm(sp_left_out_latent, right_out_latent))
# print(calc_diff_norm(left_out_latent, right_out_latent))
