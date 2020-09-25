# %%
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import ortho_group

from graspy.align import OrthogonalProcrustes, SeedlessProcrustes
from graspy.embed import AdjacencySpectralEmbed
from graspy.inference import LatentDistributionTest
from graspy.plot import pairplot
from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.visualization import adjplot

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)
    savefig(name, foldername=FNAME, save_on=True, fmt="png", dpi=300, **kws)


mg = load_metagraph("G")

pair_meta = pd.read_csv(
    "maggot_models/experiments/graph_match/outs/pair_meta.csv", index_col=0
)
pair_meta = pair_meta.sort_values(["hemisphere", "predicted_pair_id"])
mg = mg.reindex(pair_meta.index.values, use_ids=True)
mg = MetaGraph(mg.adj, pair_meta)
n_pairs = len(pair_meta) // 2
left_inds = np.arange(n_pairs)
right_inds = left_inds.copy() + n_pairs
left_mg = MetaGraph(mg.adj[np.ix_(left_inds, left_inds)], mg.meta.iloc[left_inds])
right_mg = MetaGraph(mg.adj[np.ix_(right_inds, right_inds)], mg.meta.iloc[right_inds])

assert (
    left_mg.meta["predicted_pair_id"].values
    == right_mg.meta["predicted_pair_id"].values
).all()
# %% [markdown]
# ##
adjplot(left_mg.adj, plot_type="scattermap")
adjplot(right_mg.adj, plot_type="scattermap")

#%%


n_components = 3
ase = AdjacencySpectralEmbed(n_components=n_components)
left_out_latent, left_in_latent = ase.fit_transform(pass_to_ranks(left_mg.adj))
right_out_latent, right_in_latent = ase.fit_transform(pass_to_ranks(right_mg.adj))

out_latents = np.concatenate((left_out_latent, right_out_latent))
in_latents = np.concatenate((left_in_latent, right_in_latent))
labels = np.array(["Left"] * len(left_out_latent) + ["Right"] * len(right_out_latent))
pairplot(out_latents, labels=labels, title="Out latent positions (no alignment)")
pairplot(in_latents, labels=labels, title="In latent positions (no alignment)")

#%%
initial_Q = np.eye(n_components)
sp = SeedlessProcrustes(init="custom", initial_Q=initial_Q)


# %% [markdown]
# ##

currtime = time.time()
sp.fit(left_out_latent, right_out_latent)
print(f"{time.time() - currtime} elapsed")

# %% [markdown]
# ##
sp_left_out_latent = sp.transform(left_out_latent)
sp_out_latents = np.concatenate((sp_left_out_latent, right_out_latent))
pairplot(
    sp_out_latents,
    labels=labels,
    title="Out latent positions (after seedless procrustes)",
)

# %% [markdown]
# ##

op = OrthogonalProcrustes()
op_left_out_latent = op.fit_transform(left_out_latent, right_out_latent)
op_out_latents = np.concatenate((op_left_out_latent, right_out_latent))

pairplot(
    op_out_latents,
    labels=labels,
    title="Out latent positions (after orthogonal procrustes)",
)


def calc_diff_norm(X, Y):
    return np.linalg.norm(X - Y, ord="fro")


print(calc_diff_norm(op_left_out_latent, right_out_latent))
print(calc_diff_norm(sp_left_out_latent, right_out_latent))
print(calc_diff_norm(left_out_latent, right_out_latent))


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

    sp = SeedlessProcrustes()
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

from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

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

import seaborn as sns
from src.visualization import set_theme
import matplotlib.pyplot as plt

set_theme()
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
from graspy.plot import screeplot

# screeplot(pass_to_ranks(left_mg.adj), cumulative=False, show_first=40)
from graspy.embed import select_dimension

elbow_inds, elbow_vals = select_dimension(pass_to_ranks(right_mg.adj), n_elbows=4)
sing_vals = ase.singular_values_

plt.plot(
    range(1, len(sing_vals) + 1), sing_vals,
)
plt.scatter(elbow_inds, elbow_vals, color="red", s=10, zorder=10)

