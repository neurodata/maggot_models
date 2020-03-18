# %% [markdown]
# #
import os

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


plt.style.use("seaborn-whitegrid")
sns.set_context("talk")

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

probs = np.geomspace(0.005, 0.05, 10)
n_syns = np.linspace(0, 100, 101)

rows = []
for p in probs:
    for n in n_syns:
        trans_prob = 1 - (1 - p) ** n
        rows.append({"trans_prob": trans_prob, "n_syns": n, "syn_prob": p})

data = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(
    data=data, x="n_syns", y="trans_prob", hue="syn_prob", ax=ax, legend="brief"
)
leg = ax.get_legend()
leg.texts[0].set_text("")
leg.set_bbox_to_anchor((1, 1))
leg.set_title("Synapse prob.")
ax.set_xlabel("Number of synapses")
ax.set_ylabel("Transmission prob.")

label_probs = [probs[0], probs[-1]]

vas = ["top", "bottom"]
has = ["left", "right"]
for i, p in enumerate(label_probs):
    val = data[data["syn_prob"] == p]
    row = val[val["n_syns"] == 20]
    trans_prob = row["trans_prob"]
    ax.text(20, trans_prob, f"p={p:0.3}", va=vas[i], ha=has[i])

stashfig("trans_probs_by_params_zoom")

