#%%
from src.data import load_new_left
from graspy.plot import heatmap
import matplotlib.pyplot as plt

savefig = False

left_adj, left_labels, left_full_labels = load_new_left(return_full_labels=True)
# right_adj, right_labels = load_new_right()

heatmap_kws = dict(cbar=False, font_scale=30)
fig, ax = plt.subplots(1, 2, figsize=(18, 10))
plt.style.use("seaborn-white")
heatmap(
    left_adj,
    inner_hier_labels=left_full_labels,
    outer_hier_labels=left_labels,
    ax=ax[1],
    hier_label_fontsize=5,
    **heatmap_kws,
)
heatmap(
    left_adj,
    inner_hier_labels=left_labels,
    ax=ax[0],
    hier_label_fontsize=10,
    **heatmap_kws,
)

plt.tight_layout()
if savefig:
    plt.savefig(
        "./maggot_models/reports/figures/raw_heatmaps/heatmaps.pdf",
        facecolor="w",
        format="pdf",
    )

