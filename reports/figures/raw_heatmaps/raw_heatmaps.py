#%%
from src.data import load_left, load_right
from graspy.plot import heatmap
from graspy.utils import binarize
import matplotlib.pyplot as plt

left_adj, left_labels = load_left()
right_adj, right_labels = load_right()

heatmap_kws = dict(cbar=False, font_scale=2)
fig, ax = plt.subplots(1, 2, figsize=(18, 10))
heatmap(left_adj, inner_hier_labels=left_labels, ax=ax[0], title="Left", **heatmap_kws)
heatmap(
    right_adj, inner_hier_labels=right_labels, ax=ax[1], title="Right", **heatmap_kws
)
plt.tight_layout()
plt.savefig(
    "./maggot_models/reports/figures/raw_heatmaps/heatmaps.pdf",
    facecolor="w",
    format="pdf",
)

