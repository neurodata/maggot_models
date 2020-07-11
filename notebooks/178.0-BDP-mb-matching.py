# %% [markdown]
# ##
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.match import GraphMatch
from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import adjplot

sns.set_context("talk")
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


# plotting settings
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
    "xtick.major.size": 0,
    "ytick.major.size": 0,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

# %% [markdown]
# ##


mg = load_metagraph("G")

# %% [markdown]
# ##
class1_types = ["KC", "MBON", "MBIN", "uPN", "mPN", "APL"]
class2_types = ["sens-ORN"]

meta = mg.meta
mb_meta = meta[meta["class1"].isin(class1_types) | meta["class2"].isin(class2_types)]
mb_meta = mb_meta[~mb_meta["partially_differentiated"]]
mb_meta = mb_meta[mb_meta["hemisphere"].isin(["L", "R"])]
mb_mg = mg.reindex(mb_meta.index, use_ids=True, inplace=False)
mb_mg.calculate_degrees(inplace=True)
mb_mg.meta["Total edgesum"] = -mb_mg.meta["Total edgesum"]
sizes = mb_mg.meta.groupby("class1").size()
mb_mg.meta["class1_sizes"] = -mb_mg.meta["class1"].map(sizes)
# %% [markdown]
# ##
meta = mb_mg.meta
print("n_left")
print(len(meta[meta["left"]]))
print("n_right")
print(len(meta[meta["right"]]))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
adjplot(
    pass_to_ranks(mb_mg.adj),
    meta=mb_mg.meta,
    sort_class=["hemisphere", "class1"],
    class_order=["class1_sizes"],
    item_order=["class1", "Total edgesum"],
    cbar=False,
    row_tick_pad=[0.05, 0.7],
    col_tick_pad=[0.2, 0.7],
    tick_rot=90,
    tick_fontsize=12,
    gridline_kws=dict(color="grey", linestyle="--", linewidth=0.5),
    ax=ax,
)

# %% [markdown]
# ##
left_mb_mg = mb_mg.reindex(meta[meta["left"]].index, use_ids=True, inplace=False)
right_mb_mg = mb_mg.reindex(meta[meta["right"]].index, use_ids=True, inplace=False)

# %% [markdown]
# ##
assert (
    np.unique(left_mb_mg.meta["pair_id"]) == np.unique(right_mb_mg.meta["pair_id"])
).all()

# %% [markdown]
# ##


meta = mb_mg.meta
meta["inds"] = range(len(meta))
left_inds, right_inds = get_paired_inds(meta)
left_adj = mb_mg.adj[np.ix_(left_inds, left_inds)]
right_adj = mb_mg.adj[np.ix_(right_inds, right_inds)]
# shuffle_inds = np.random.choice(len(right_adj), len(right_adj), False)

np.random.seed(8888)
rows = []
n_inits = 256
for i in range(n_inits):
    for ptr in [True, False]:
        gm = GraphMatch(n_init=1, init_method="rand", eps=0.9, shuffle_input=False)
        gm.fit(pass_to_ranks(left_adj), pass_to_ranks(right_adj))
        right_meta = mb_mg.meta.iloc[right_inds.values[gm.perm_inds_]]
        left_meta = mb_mg.meta.iloc[left_inds]
        match_ratio = (
            left_meta["pair_id"].values == right_meta["pair_id"].values
        ).sum() / len(left_inds)
        rows.append(
            dict(
                ptr=ptr,
                init=i,
                match_ratio=match_ratio,
                score=gm.score_,
                perm_inds=gm.perm_inds_,
            )
        )
# %% [markdown]
# ##
results = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
order = [False, True]
sns.stripplot(
    data=results, x="ptr", y="match_ratio", alpha=0.3, jitter=0.4, order=order
)
maxs = results.groupby("ptr")["match_ratio"].max()
for i, item in enumerate(order):
    ax.text(
        i,
        maxs[item] + 0.01,
        f"{maxs[item]:.3f}",
        va="bottom",
        ha="center",
        color="black",
    )
stashfig("match_ratio_by_method")

# %% [markdown]
# ##
colors = sns.color_palette("deep", n_colors=3, desat=0.9)
palette = {True: colors[0], False: colors[1]}
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(
    data=results,
    x="score",
    y="match_ratio",
    s=15,
    alpha=0.7,
    ax=ax,
    hue="ptr",
    palette=palette,
)
best_inds = results.groupby("ptr")["score"].idxmax()
best_results = results.loc[best_inds]
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for idx, row in best_results.iterrows():
    ptr = row["ptr"]
    color = palette[ptr]
    match_ratio = row["match_ratio"]
    score = row["score"]
    ax.plot(
        [score, score, xlim[0]],
        [ylim[0], match_ratio, match_ratio],
        color=color,
        linestyle="--",
        linewidth=1,
    )
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
stashfig("mb-score-vs-acc")
# %% [markdown]
# ##
ptr_results = results[results["ptr"]]
raw_results = results[~results["ptr"]]
print(ptr_results.loc[ptr_results["score"].idxmax()])
print(raw_results.loc[raw_results["score"].idxmax()])

# %% [markdown]
# ##

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# adjplot(left_adj, ax=axs[0], cbar=False)
# adjplot(right_adj[np.ix_(gm.perm_inds_, gm.perm_inds_)], ax=axs[1], cbar=False)

# %% [markdown]
# ## select the perm inds to use
best_idx = results["score"].idxmax()
perm_inds = results.loc[best_idx, "perm_inds"]
meta = mb_mg.meta
left_meta = meta.iloc[left_inds].copy()
left_meta.reset_index(inplace=True)
right_meta = meta.iloc[right_inds].copy()
right_meta = right_meta.iloc[perm_inds]
right_meta.reset_index(inplace=True)
left_meta["predicted_pair_id"] = range(len(left_meta))
right_meta["predicted_pair_id"] = range(len(right_meta))
correct_match = left_meta["pair_id"] == right_meta["pair_id"]
left_meta = left_meta.rename(columns=lambda x: "left_" + x)
right_meta = right_meta.rename(columns=lambda x: "right_" + x)
joint_meta = pd.concat((left_meta, right_meta), axis=1)
joint_meta["correct_match"] = correct_match
joint_meta["class1_sizes"] = joint_meta["left_class1"].map(
    -joint_meta.groupby("left_class1").size()
)
# %% [markdown]
# ##
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

palette = {True: "#4daf4a", False: "#ff7f00"}
colors = sns.color_palette("deep", n_colors=3, desat=0.9)
palette = {True: colors[2], False: colors[1]}
adjplot_kws = dict(
    cbar=False,
    meta=joint_meta,
    sort_class=["left_class1"],
    class_order=["class1_sizes"],
    item_order=["correct_match", "left_Total edgesum"],
    colors=["correct_match"],
    palette=palette,
    tick_rot=90,
    gridline_kws=dict(color="grey", linestyle="--", linewidth=0.5),
)
ax = axs[0]
adjplot(pass_to_ranks(left_adj), ax=ax, title="Left", **adjplot_kws)
ax = axs[1]
right_sort_adj = right_adj.copy()
right_sort_adj = right_sort_adj[np.ix_(gm.perm_inds_, gm.perm_inds_)]
adjplot(pass_to_ranks(right_sort_adj), ax=ax, title="Right", **adjplot_kws)
stashfig("left-right-predicted-adj", fmt="pdf")
# %% [markdown]
# ##

from graspy.embed import AdjacencySpectralEmbed
from umap import UMAP
from src.visualization import CLASS_COLOR_DICT

ase = AdjacencySpectralEmbed(n_components=3, check_lcc=False)
embed = ase.fit_transform(pass_to_ranks(left_adj))
embed = np.concatenate(embed, axis=1)

umap_embed = UMAP(n_neighbors=5, min_dist=1).fit_transform(embed)

plot_df = pd.DataFrame(data=umap_embed)
plot_df.index = mb_mg.meta.iloc[left_inds].index
plot_df = pd.concat((plot_df, mb_mg.meta.iloc[left_inds]), axis=1)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
)

# sns.scatterplot(x=umap_embed[:, 0], y=umap_embed[:, 1])
# %% [markdown]
# ##
match_df = pd.read_csv("maggot_models/notebooks/matched_metadata.csv", index_col=0)
paired_df = match_df[match_df["pair_id"] != -1]
print((paired_df["pair"] == paired_df["predicted pair"]).mean())

