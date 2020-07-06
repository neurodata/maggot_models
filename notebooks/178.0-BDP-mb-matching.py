# %% [markdown]
# ##
from src.io import savefig, savecsv
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

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

from src.data import load_metagraph

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

# print(len(meta[meta["hemisphere"] == "center"]))
from src.visualization import adjplot
from graspy.utils import pass_to_ranks

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
# plt.setp(ax.get_ylabel(), fontsize=2)

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
from graspy.match import GraphMatch


from src.utils import get_paired_inds

meta = mb_mg.meta
meta["inds"] = range(len(meta))
left_inds, right_inds = get_paired_inds(meta)
left_adj = mb_mg.adj[np.ix_(left_inds, left_inds)]
right_adj = mb_mg.adj[np.ix_(right_inds, right_inds)]
shuffle_inds = np.random.choice(len(right_adj), len(right_adj), False)
# right_adj = right_adj[shuffle_inds]

rows = []
n_inits = [1, 2, 4, 8, 64]
for n_init in n_inits:
    for ptr in [True, False]:
        gm = GraphMatch(n_init=n_init, init_method="rand", eps=0.1, shuffle_input=False)
        gm.fit(pass_to_ranks(left_adj), pass_to_ranks(right_adj))
        right_meta = mb_mg.meta.iloc[right_inds.values[gm.perm_inds_]]
        left_meta = mb_mg.meta.iloc[left_inds]
        match_ratio = (
            left_meta["pair_id"].values == right_meta["pair_id"].values
        ).sum() / len(left_inds)
        rows.append(dict(ptr=ptr, n_init=n_init, match_ratio=match_ratio))

import pandas as pd

# %% [markdown]
# ##
results = pd.DataFrame(rows)
sns.lineplot(x="n_init", y="match_ratio", hue="ptr", data=results)

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# adjplot(left_adj, ax=axs[0], cbar=False)
# adjplot(right_adj[np.ix_(gm.perm_inds_, gm.perm_inds_)], ax=axs[1], cbar=False)



# %%
