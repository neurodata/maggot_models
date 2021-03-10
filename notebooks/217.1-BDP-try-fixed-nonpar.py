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
from graspologic.utils import get_multigraph_intersect_lcc

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
mg = mg.remove_pdiff()
mg = mg.make_lcc()
mg.meta["inds"] = range(len(mg))
left_inds, right_inds = get_paired_inds(mg.meta)
n_pairs = len(left_inds)
left_mg = MetaGraph(mg.adj[np.ix_(left_inds, left_inds)], mg.meta.iloc[left_inds])
right_mg = MetaGraph(mg.adj[np.ix_(right_inds, right_inds)], mg.meta.iloc[right_inds])
left_adj = left_mg.adj
right_adj = right_mg.adj

adjs, lcc_inds = get_multigraph_intersect_lcc([left_adj, right_adj], return_inds=True)
left_adj = adjs[0]
right_adj = adjs[1]
n_pairs = len(left_inds)

print(f"Working with {n_pairs} pairs.")

#%%[markdown]
### Plot the adjacency matrices, sorted by known pairs
# %%
# collapse
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
adjplot(
    left_adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[0],
    title=r"Left $\to$ left",
    color=palette["Left"],
)
adjplot(
    right_adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[1],
    title=r"Right $\to$ right",
)
stashfig("left-right-adjs")

#%% [markdown]
### Embed the ipsilateral subgraphs

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


def embed(adj, n_components=40, ptr=False):
    if ptr:
        adj = pass_to_ranks(adj)
    elbow_inds, elbow_vals = select_dimension(augment_diagonal(adj), n_elbows=4)
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(adj)
    return out_latent, in_latent, ase.singular_values_, elbow_inds


n_components = 8
max_n_components = 40
preprocess = "binarize"

if preprocess == "binarize":
    left_adj = binarize(left_adj)
    right_adj = binarize(right_adj)

left_out_latent, left_in_latent, left_sing_vals, left_elbow_inds = embed(
    left_adj, n_components=max_n_components
)
right_out_latent, right_in_latent, right_sing_vals, right_elbow_inds = embed(
    right_adj, n_components=max_n_components
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
#%%
print(np.linalg.norm(binarize(left_mg.adj)))
print(np.linalg.norm(binarize(right_mg.adj)))

#%% [markdown]
### Align the embeddings
# Here I align first using Procrustes, and then using Seedless Procrustes initialized at
# the Procrustes solution. I will refer to this as "oracle initialization".
# %%
# collapse


def run_alignments(X, Y, scale=False):
    X = X.copy()
    Y = Y.copy()
    if scale:
        X_norm = np.linalg.norm(X, ord="fro")
        Y_norm = np.linalg.norm(Y, ord="fro")
        avg_norms = (X_norm + Y_norm) / 2
        X = X * (avg_norms / X_norm)
        Y = Y * (avg_norms / Y_norm)
    op = OrthogonalProcrustes()
    X_trans_op = op.fit_transform(X, Y)
    sp = SeedlessProcrustes(init="custom", initial_Q=op.Q_)
    X_trans_sp = sp.fit_transform(X, Y)
    return X_trans_op, X_trans_sp


def calc_diff_norm(X, Y):
    return np.linalg.norm(X - Y, ord="fro")


n_components = 8
op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent[:, :n_components], right_out_latent[:, :n_components]
)
op_diff_norm = calc_diff_norm(op_left_out_latent, right_out_latent[:, :n_components])
sp_diff_norm = calc_diff_norm(sp_left_out_latent, right_out_latent[:, :n_components])

print(f"Procrustes diff. norm using true pairs: {op_diff_norm}")
print(f"Seedless Procrustes diff. norm using true pairs: {sp_diff_norm}")


#%%
# collapse

plot_latents(
    op_left_out_latent,
    right_out_latent[:, :n_components],
    "Out latent positions (Procrustes alignment)",
    n_show=8,
)
stashfig(f"out-latent-procrustes-preprocess={preprocess}")

plot_latents(
    sp_left_out_latent,
    right_out_latent[:, :n_components],
    "Out latent positions (Seedless Procrustes alignment, oracle init)",
    n_show=8,
)
stashfig(f"out-latent-seedless-oracle-preprocess={preprocess}")

#%% [markdown]
# ## Defining the test to do
# Distance correlation, 2-sample test, 500 bootstraps
#%%
test = "dcorr"
n_bootstraps = 500
workers = -1
auto = False


def run_test(
    X1,
    X2,
    rows=None,
    info={},
    auto=auto,
    n_bootstraps=n_bootstraps,
    workers=workers,
    test=test,
):
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
    if rows is not None:
        rows.append(row)
    else:
        return row


#%% [markdown]
# ## Drill down on the d=1 case

#%% [markdown]
# ### Setting up some metadata
#%%
left_arange = np.arange(n_pairs)
right_arange = left_arange + n_pairs
embedding_1d_df = pd.DataFrame(index=np.arange(2 * n_pairs))
embedding_1d_df["pair_ind"] = np.concatenate((left_arange, left_arange))
embedding_1d_df["hemisphere"] = "Right"
embedding_1d_df.loc[left_arange, "hemisphere"] = "Left"
embedding_1d_df["x"] = 1
embedding_1d_df.loc[left_arange, "x"] = 0

#%% [markdown]
# ### Align and test the d=1, out embedding
#%%
n_components = 1
op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent[:, :n_components], right_out_latent[:, :n_components]
)

from scipy.stats import ks_2samp

ks_2samp(op_left_out_latent[:, 0], right_out_latent[:, 0])

#%%
from scipy.stats import special_ortho_group

n_components = 8
alpha = 0.95
W = alpha * np.eye(n_components) + (1 - alpha) * special_ortho_group.rvs(n_components)

#%%
sns.heatmap(W @ W.T)
#%%
np.abs(np.linalg.eigvals(W))
# embedding_1d_df.loc[left_arange, "out_1d_align"] = op_left_out_latent[:, 0]
# embedding_1d_df.loc[right_arange, "out_1d_align"] = right_out_latent[:, 0]

# test_results_1d_align = run_test(
#     op_left_out_latent[:, :1], right_out_latent[:, :1], auto=False
# )

#%% [markdown]
# ### Align the d=8 out embedding, test on the first dimension of that
n_components = 8
op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent[:, :n_components], right_out_latent[:, :n_components]
)

embedding_1d_df.loc[left_arange, "out_8d_align"] = op_left_out_latent[:, 0]
embedding_1d_df.loc[right_arange, "out_8d_align"] = right_out_latent[:, 0]

test_results_8d_align = run_test(
    op_left_out_latent[:, :1], right_out_latent[:, :1], auto=False
)

#%% [markdown]
# ### Plot the results
#%%

fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
histplot_kws = dict(
    stat="density",
    cumulative=True,
    element="poly",
    common_norm=False,
    bins=np.linspace(0, 1, 1000),
    fill=False,
)


ax = axs[0]
sns.histplot(
    data=embedding_1d_df,
    x="out_1d_align",
    ax=ax,
    hue="hemisphere",
    legend=False,
    palette=palette,
    **histplot_kws,
)
# ax.text(
#     0.8,
#     0.8,
#     ,
#     transform=ax.transAxes,
#     color="black",
#     ha="right",
# )
ax.set_title(f"OP d=1 alignment, p-value:\n {test_results_1d_align['pvalue']:0.3f}")

ax = axs[1]
sns.histplot(
    data=embedding_1d_df,
    x="out_8d_align",
    ax=ax,
    hue="hemisphere",
    palette=palette,
    **histplot_kws,
)
# ax.text(
#     0.8,
#     0.8,
#     f"2S-Dcorr p-value:\n {test_results_8d_align['pvalue']:0.3f}",
#     transform=ax.transAxes,
#     color="black",
#     ha="right",
# )
handles, labels = ax.get_legend_handles_labels()
# ax.get_legend().remove()
from matplotlib.lines import Line2D

left_line = Line2D([0, 1], [0, 1], color=palette["Left"], label="Left")
right_line = Line2D([0, 1], [0, 1], color=palette["Right"], label="Right")
ax.set_title(f"OP d=8 alignment, p-value:\n {test_results_8d_align['pvalue']:0.3f}")
stashfig(f"dim1-focus-test={test}-n_bootstraps={n_bootstraps}-preprocess={preprocess}")
#%%

#%%
fig, ax = plt.subplots(1, 1, figsize=(2, 6))
sns.lineplot(
    data=embedding_1d_df,
    x="x",
    y="out_1d",
    hue="pair_ind",
    linewidth=0.5,
    alpha=0.3,
    legend=False,
)

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
for n_components in np.arange(1, 11):
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
