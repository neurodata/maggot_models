#%%
from src.data import load_maggot_graph

mg = load_maggot_graph()

nodes = mg.nodes.query("selected_lcc")

#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualization import set_theme

set_theme()

sfs = (
    nodes.groupby("dc_level_7_n_components=10_min_split=32")["sum_signal_flow"]
    .mean()
    .sort_values(ascending=False)
)
sf_ranks = pd.Series(index=sfs.index, data=np.arange(len(sfs)))

wss = (
    nodes.groupby("dc_level_7_n_components=10_min_split=32")["sum_walk_sort"]
    .mean()
    .sort_values(ascending=True)
)
ws_ranks = pd.Series(index=wss.index, data=np.arange(len(wss)))

sizes = (
    nodes.groupby("dc_level_7_n_components=10_min_split=32").size().loc[ws_ranks.index]
)
sizes.name = "# in cluster"

sf_ranks = sf_ranks.loc[ws_ranks.index]

fig, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=sf_ranks, y=ws_ranks, size=sizes, ax=ax)
ax.set(xlabel="Signal Flow Rank", ylabel="Walk Sort Rank")
fig.set_facecolor("w")

from scipy.stats import pearsonr, spearmanr


# pearsonr(sf_ranks, ws_ranks)
rho, p = spearmanr(sf_ranks, ws_ranks)

ax.text(0.6, 0.1, r"Spearman's $\rho$ = " + f"{rho:.2f}", transform=ax.transAxes)
