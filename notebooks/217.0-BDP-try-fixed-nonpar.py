#%% [markdown]
## Maggot embedding alignment
# > Investigating alignments of the maggot data embeddings
#
# - toc: false
# - badges: false
# - categories: [pedigo, graspologic, maggot]
# - hide: false
# - search_exclude: false
# %%
# collapse
import os
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyppo.ksample import KSample
from sklearn.neighbors import NearestNeighbors

import graspologic as gl
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed, select_dimension
from graspologic.plot import pairplot
from graspologic.utils import augment_diagonal, binarize, pass_to_ranks
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.utils import get_paired_inds
from src.visualization import CLASS_COLOR_DICT, adjplot, set_theme

data_version = "2020-09-23"
print(f"graspologic version: {gl.__version__}")
print(f"seaborn version: {sns.__version__}")

print(f"data version: {data_version}")

set_theme()
colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right", "OP", "O-SP"], colors))

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="pdf", **kws)
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=300, **kws)


#%% [markdown]
### Load in the data, use only the known pairs
# %%
# collapse


mg = load_metagraph("G", version=data_version)
mg = mg.make_lcc()
mg.meta["inds"] = range(len(mg))
left_inds, right_inds = get_paired_inds(mg.meta)
n_pairs = len(left_inds)
left_mg = MetaGraph(mg.adj[np.ix_(left_inds, left_inds)], mg.meta.iloc[left_inds])
right_mg = MetaGraph(mg.adj[np.ix_(right_inds, right_inds)], mg.meta.iloc[right_inds])

print(f"Working with {n_pairs} pairs.")

#%%[markdown]
### Plot the adjacency matrices, sorted by known pairs
# %%
# collapse
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

#%% [markdown]
### Embed the ipsilateral subgraphs
# - Computations are happening in `n_components` dimensions, where `n_components = 8` (dashed line in screeplot)
# - In many plots, I only show the first 4 dimensions just for clarity
#%%
# collapse


def plot_latents(left, right, title="", n_show=4):
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(plot_data[:, :n_show], labels=labels, title=title)
    return pg


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    plt.scatter(
        elbow_inds, sing_vals[elbow_inds - 1], marker="x", s=50, zorder=10, color=color
    )
    ax.set(ylabel="Singular value", xlabel="Index")
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
preprocess = "binarize"

left_adj = left_mg.adj
right_adj = right_mg.adj
if preprocess == "binarize":
    left_adj = binarize(left_adj)
    right_adj = binarize(right_adj)

left_out_latent, left_in_latent, left_sing_vals, left_elbow_inds = embed(
    binarize(left_mg.adj), n_components=max_n_components
)
right_out_latent, right_in_latent, right_sing_vals, right_elbow_inds = embed(
    binarize(right_mg.adj), n_components=max_n_components
)

# plot the screeplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(left_sing_vals, left_elbow_inds, color=palette["Left"], ax=ax, label="Left")
screeplot(
    right_sing_vals, right_elbow_inds, color=palette["Right"], ax=ax, label="Right"
)
ax.legend()
ax.axvline(n_components, color="black", linewidth=1.5, linestyle="--")
stashfig(f"screeplot-preprocess={preprocess}")

# plot the latent positions before any kind of alignment
plot_latents(
    left_out_latent, right_out_latent, title="Out latent positions (no alignment)"
)
stashfig(f"out-latent-no-align-preprocess={preprocess}")
plot_latents(
    left_in_latent, right_in_latent, title="In latent positions (no alignment)"
)
stashfig(f"in-latent-no-align-preprocess={preprocess}")

#%% [markdown]
### Align the embeddings
# Here I align first using Procrustes, and then using Seedless Procrustes initialized at
# the Procrustes solution. I will refer to this as "oracle initialization".
# %%
# collapse


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
print(f"Seedless Procrustes diff. norm using true pairs: {sp_diff_norm}")

#%%


#%%
# collapse

plot_latents(
    op_left_out_latent,
    right_out_latent,
    "Out latent positions (Procrustes alignment)",
    n_show=8,
)
stashfig(f"out-latent-procrustes-preprocess={preprocess}")

plot_latents(
    sp_left_out_latent,
    right_out_latent,
    "Out latent positions (Seedless Procrustes alignment, oracle init)",
    n_show=8,
)
stashfig(f"out-latent-seedless-oracle-preprocess={preprocess}")


# %%
# collapse

op_left_in_latent, sp_left_in_latent = run_alignments(left_in_latent, right_in_latent)

#%%[markdown]
### Looking at nearest neighbors in the aligned embeddings
# - I am concatenating the in and out embeddings *after* learning the alignments separately for each
# - "P by K" (short for precision by K) is the proportion of nodes for which their true pair is
# within its K nearest neighbors
# - I plot the cumulative density of the measure described above. So for a point `x, y` on the plot below,
# the result can be read as "`y` of all nodes have their true pair within their `x` nearest neighbors in
# the aligned embedded space".
# - I perform the experiment above using both orthogonal Procrustes (OP) and Seedless Procrustes (SP) with oracale initialization
# %%
# collapse


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
    (op_left_out_latent[:, :n_components], op_left_in_latent[:, :n_components]), axis=1
)
sp_left_composite_latent = np.concatenate(
    (sp_left_out_latent[:, :n_components], sp_left_in_latent[:, :n_components]), axis=1
)
right_composite_latent = np.concatenate(
    (right_out_latent[:, :n_components], right_in_latent[:, :n_components]), axis=1
)


op_ranks = calc_nn_ranks(op_left_composite_latent, right_composite_latent)
sp_ranks = calc_nn_ranks(sp_left_composite_latent, right_composite_latent)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_rank_cdf(op_ranks, color=palette["OP"], ax=ax, label="OP")
plot_rank_cdf(sp_ranks, color=palette["O-SP"], ax=ax, label="O-SP")
ax.legend()
stashfig(f"rank-cdf-n_components={n_components}-preprocess={preprocess}")

#%%
# collapse
test = "dcorr"
n_bootstraps = 500
workers = -1
auto = False


def run_test(X1, X2, rows, info={}):
    currtime = time.time()
    test_obj = KSample(test)

    tstat, pvalue = test_obj.test(
        X1,
        X2,
        reps=n_bootstraps,
        workers=workers,
        auto=auto,
    )
    elapsed = time.time() - currtime
    row = {
        "pvalue": pvalue,
        "tstat": tstat,
        "elapsed": elapsed,
    }
    row.update(info)
    pprint.pprint(row)
    rows.append(row)


rows = []
for n_components in np.arange(1, 2):
    op_left_out_latent, sp_left_out_latent = run_alignments(
        left_out_latent[:, :n_components], right_out_latent[:, :n_components]
    )
    op_left_in_latent, sp_left_in_latent = run_alignments(
        left_in_latent[:, :n_components], right_in_latent[:, :n_components]
    )
    op_left_composite_latent = np.concatenate(
        (op_left_out_latent, op_left_in_latent), axis=1
    )
    sp_left_composite_latent = np.concatenate(
        (sp_left_out_latent, sp_left_in_latent), axis=1
    )
    right_composite_latent = np.concatenate(
        (right_out_latent[:, :n_components], right_in_latent[:, :n_components]), axis=1
    )

    run_test(
        op_left_composite_latent,
        right_composite_latent,
        rows,
        info={"alignment": "op", "n_components": n_components},
    )
    run_test(
        sp_left_composite_latent,
        right_composite_latent,
        rows,
        info={"alignment": "o-sp", "n_components": n_components},
    )

#%%
# collapse
colors = sns.color_palette("Set1")
pal = {"op": colors[2], "o-sp": colors[3]}
results = pd.DataFrame(rows)
results["n_components"] += np.random.uniform(-0.2, 0.2, size=len(results))
results["log10(pvalue)"] = np.log10(results["pvalue"])
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[0]
sns.scatterplot(
    data=results,
    x="n_components",
    y="pvalue",
    hue="alignment",
    palette=pal,
    ax=ax,
    s=40,
)
ax.get_legend().remove()
ax.axhline(0.05, color="black", alpha=0.7, linewidth=1.5, zorder=-1)
ax.axhline(0.005, color="black", alpha=0.7, linewidth=1.5, linestyle="--", zorder=-1)
ax.axhline(
    1 / n_bootstraps, color="black", alpha=0.7, linewidth=1.5, linestyle=":", zorder=-1
)

ax = axs[1]
sns.scatterplot(
    data=results,
    x="n_components",
    y="log10(pvalue)",
    hue="alignment",
    palette=pal,
    ax=ax,
    s=40,
)
ax.axhline(np.log10(0.05), color="black", alpha=0.7, linewidth=1.5, zorder=-1)
ax.text(ax.get_xlim()[-1] + 0.1, np.log10(0.05), 0.05, ha="left", va="center")
ax.axhline(
    np.log10(0.005), color="black", alpha=0.7, linewidth=1.5, linestyle="--", zorder=-1
)
ax.text(
    ax.get_xlim()[-1] + 0.1,
    np.log10(0.005),
    "0.005",
    ha="left",
    va="center",
)
ax.axhline(
    np.log10(1 / n_bootstraps),
    color="black",
    alpha=0.7,
    linewidth=1.5,
    linestyle=":",
    zorder=-1,
)
ax.text(
    ax.get_xlim()[-1] + 0.1,
    np.log10(1 / n_bootstraps),
    r"0.002 = $\frac{1}{n_{bootstraps}}$",
    ha="left",
    va="center",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="alignment")
plt.tight_layout()
stashfig(
    f"nonpar-pvalues-test={test}-n_bootstraps={n_bootstraps}-preprocess={preprocess}"
)

#%%
from graspologic.utils import symmetrize

evals = np.linalg.eigvals(symmetrize(left_adj))
sort_inds = np.argsort(np.abs(evals))[::-1]
evals[sort_inds][:10]

#%%
# need to look at why some of them are popping up as significant when d=8 is not,
# for example.

n_components = 8

X = left_out_latent[:, :n_components]
Y = right_out_latent[:, :n_components]
op = OrthogonalProcrustes()
rot_left_out_latent = op.fit_transform(X, Y)
Q_out = op.Q_

X = left_in_latent[:, :n_components]
Y = right_in_latent[:, :n_components]
op = OrthogonalProcrustes()
rot_left_in_latent = op.fit_transform(X, Y)
Q_in = op.Q_

d8_left_composite_latent = np.concatenate(
    (rot_left_out_latent, rot_left_in_latent), axis=1
)

d8_right_composite_latent = np.concatenate(
    (right_out_latent[:, :n_components], right_in_latent[:, :n_components])
)
plot_latents(
    rot_left_out_latent,
    right_out_latent[:, :n_components],
    "D=8, out latent positions",
    n_show=8,
)
#%%
Q_out_d9 = np.zeros((9, 9))
Q_out_d9[:8, :8] = Q_out
Q_out_d9[8, 8] = 1

Q_in_d9 = np.zeros((9, 9))
Q_in_d9[:8, :8] = Q_in
Q_in_d9[8, 8] = 1

n_components = 9

rot_left_out_latent = left_out_latent[:, :n_components] @ Q_out_d9

rot_left_in_latent = left_in_latent[:, :n_components] @ Q_in_d9

d9_left_composite_latent = np.concatenate(
    (rot_left_out_latent, rot_left_in_latent), axis=1
)

d9_right_composite_latent = np.concatenate(
    (right_out_latent[:, :n_components], right_in_latent[:, :n_components])
)
#%%

plot_latents(
    rot_left_out_latent,
    right_out_latent[:, :n_components],
    "D=9, out latent positions",
    n_show=n_components,
)
#%%
focus_rows = []
run_test(
    d8_left_composite_latent,
    d8_right_composite_latent,
    focus_rows,
    info={"n_components": 8},
)
run_test(
    d9_left_composite_latent,
    d9_right_composite_latent,
    focus_rows,
    info={"n_components": 9},
)

# op_left_out_latent, sp_left_out_latent = run_alignments(
#     left_out_latent[:, :n_components], right_out_latent[:, :n_components]
# )
# op_left_in_latent, sp_left_in_latent = run_alignments(
#     left_in_latent[:, :n_components], right_in_latent[:, :n_components]
# )
# op_left_composite_latent = np.concatenate(
#     (op_left_out_latent, op_left_in_latent), axis=1
# )
# sp_left_composite_latent = np.concatenate(
#     (sp_left_out_latent, sp_left_in_latent), axis=1
# )
# right_composite_latent = np.concatenate(
#     (right_out_latent[:, :n_components], right_in_latent[:, :n_components]), axis=1
# )


#%%
n_components = 8
n_show = 6
op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent[:, :n_components], right_out_latent[:, :n_components]
)
print(calc_diff_norm(op_left_out_latent[:, :n_show], right_out_latent[:, :n_show]))
plot_latents(
    op_left_out_latent,
    right_out_latent[:, :n_components],
    n_show=8,
    title="Out latent positions, OP learned on 8 dimensions",
)
stashfig("pairs-op-out-learned-on-8")

n_components = 6
op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent[:, :n_components], right_out_latent[:, :n_components]
)
print(calc_diff_norm(op_left_out_latent[:, :n_show], right_out_latent[:, :n_show]))
plot_latents(
    op_left_out_latent,
    right_out_latent[:, :n_components],
    n_show=6,
    title="Out latent positions, OP learned on 6 dimensions",
)
stashfig("pairs-op-out-learned-on-6")

# %%
n_components = 8
X = left_out_latent[:, :n_components]
Y = right_out_latent[:, :n_components]
op = OrthogonalProcrustes()
rot_left_out_latent = op.fit_transform(X, Y)
d8_Q_out = op.Q_

plot_latents(
    rot_left_out_latent,
    right_out_latent[:, :n_components],
    "D=8, out latent positions",
    n_show=8,
)
#%%
d6_Q_out = d8_Q_out[:6, :6]

d6_rot_left_out_latent = left_out_latent[:, :6] @ d6_Q_out
print(calc_diff_norm(d6_rot_left_out_latent, right_out_latent[:, :6]))

plot_latents(
    d6_rot_left_out_latent,
    right_out_latent[:, :6],
    "D=6, out latent positions OP-d8",
    n_show=6,
)


# %%


def run_test(X1, X2, rows, info={}):
    currtime = time.time()
    test_obj = KSample(test)

    tstat, pvalue = test_obj.test(
        X1,
        X2,
        reps=n_bootstraps,
        workers=workers,
        auto=auto,
    )
    elapsed = time.time() - currtime
    row = {
        "pvalue": pvalue,
        "tstat": tstat,
        "elapsed": elapsed,
    }
    row.update(info)
    pprint.pprint(row)
    rows.append(row)


align_n_components = 8
op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent[:, :align_n_components], right_out_latent[:, :align_n_components]
)
op_left_in_latent, sp_left_in_latent = run_alignments(
    left_in_latent[:, :align_n_components], right_in_latent[:, :align_n_components]
)
focus_rows = []
for n_components in np.arange(1, align_n_components + 1):
    left_out = op_left_out_latent.copy()[:, :n_components]
    left_in = op_left_in_latent.copy()[:, :n_components]
    right_out = right_out_latent[:, :align_n_components].copy()[:, :n_components]
    right_in = right_in_latent[:, :align_n_components].copy()[:, :n_components]

    # left_out[:, n_components:] = 0
    # right_out[:, n_components:] = 0
    # left_out[:, n_components:] = 0
    # left_out[:, n_components:] = 0

    left_composite_latent = np.concatenate((left_out, left_in), axis=1)
    right_composite_latent = np.concatenate((right_out, right_in), axis=1)

    run_test(
        left_composite_latent,
        right_composite_latent,
        focus_rows,
        info={"alignment": "op", "n_components": n_components},
    )

# %%
results = pd.DataFrame(focus_rows)
results["n_components"] += np.random.uniform(-0.2, 0.2, size=len(results))
results["log10(pvalue)"] = np.log10(results["pvalue"])
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[0]
sns.scatterplot(
    data=results,
    x="n_components",
    y="pvalue",
    hue="alignment",
    palette=pal,
    ax=ax,
    s=40,
)
ax.get_legend().remove()
ax.axhline(0.05, color="black", alpha=0.7, linewidth=1.5, zorder=-1)
ax.axhline(0.005, color="black", alpha=0.7, linewidth=1.5, linestyle="--", zorder=-1)
ax.axhline(
    1 / n_bootstraps, color="black", alpha=0.7, linewidth=1.5, linestyle=":", zorder=-1
)

ax = axs[1]
sns.scatterplot(
    data=results,
    x="n_components",
    y="log10(pvalue)",
    hue="alignment",
    palette=pal,
    ax=ax,
    s=40,
)
ax.axhline(np.log10(0.05), color="black", alpha=0.7, linewidth=1.5, zorder=-1)
ax.text(ax.get_xlim()[-1] + 0.1, np.log10(0.05), 0.05, ha="left", va="center")
ax.axhline(
    np.log10(0.005), color="black", alpha=0.7, linewidth=1.5, linestyle="--", zorder=-1
)
ax.text(
    ax.get_xlim()[-1] + 0.1,
    np.log10(0.005),
    "0.005",
    ha="left",
    va="center",
)
ax.axhline(
    np.log10(1 / n_bootstraps),
    color="black",
    alpha=0.7,
    linewidth=1.5,
    linestyle=":",
    zorder=-1,
)
ax.text(
    ax.get_xlim()[-1] + 0.1,
    np.log10(1 / n_bootstraps),
    r"0.002 = $\frac{1}{n_{bootstraps}}$",
    ha="left",
    va="center",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="alignment")
plt.tight_layout()
stashfig(
    f"modified-nonpar-pvalues-test={test}-n_bootstraps={n_bootstraps}-preprocess={preprocess}"
)

#%%
# look at the 1d and the 2 embeddings, see if they're the same or what is going on here
