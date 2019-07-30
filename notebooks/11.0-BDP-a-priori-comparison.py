#%%
import numpy as np
import pandas as pd

import src.utils as utils
from src.utils import get_best
from src.data import load_left, load_right
from src.models import (
    fit_a_priori,
    select_dcsbm,
    select_rdpg,
    select_sbm,
    GridSweep,
    gen_scorers,
)
from graspy.plot import heatmap
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
import seaborn as sns

# Vertex agnostic
#
#

# Load SBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_sbm"
run = 2
config = utils.load_config(base_path, experiment, run)
sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

# Load a priori
experiment = "fit_a_priori"
run = 4
config = utils.load_config(base_path, experiment, run)
ap_sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
ap_sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

# Load opposites
experiment = "fit_opposite_a_priori"
run = 4
config = utils.load_config(base_path, experiment, run)
right_pred_sbm_df = utils.load_pickle(base_path, experiment, run, "right_pred_sbm_df")
left_pred_sbm_df = utils.load_pickle(base_path, experiment, run, "left_pred_sbm_df")

# Vertex aware
#
#

# Load DCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 9  # 8
config = utils.load_config(base_path, experiment, run)
dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

# Load dDCSBM
experiment = "fit_ddcsbm"
run = 4
config = utils.load_config(base_path, experiment, run)
ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

# Load RDPG
experiment = "fit_rdpg"
run = 4
config = utils.load_config(base_path, experiment, run)
rdpg_left_df = utils.load_pickle(base_path, experiment, run, "rdpg_left_df")
rdpg_right_df = utils.load_pickle(base_path, experiment, run, "rdpg_right_df")

# Load a priori
experiment = "fit_a_priori"
run = 4
config = utils.load_config(base_path, experiment, run)
ap_dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
ap_dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

ap_ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ap_ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")


score_name = "sc_likelihood"
best_sbm_left_df = get_best(
    sbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_sbm_right_df = get_best(
    sbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_dcsbm_left_df = get_best(
    dcsbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_dcsbm_right_df = get_best(
    dcsbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_ddcsbm_left_df = get_best(
    ddcsbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_ddcsbm_right_df = get_best(
    ddcsbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_rdpg_left_df = get_best(
    rdpg_left_df, param_name="param_n_components", score_name=score_name
)
best_rdpg_right_df = get_best(
    rdpg_right_df, param_name="param_n_components", score_name=score_name
)


#%%
from pathlib import Path
import matplotlib.pyplot as plt


def save(name, fmt="pdf"):
    path = Path("/Users/bpedigo/JHU_code/maggot_models/maggot_models/notebooks/outs")
    plt.savefig(path / str(name + "." + fmt), fmt=fmt, facecolor="w")


plt.style.use("seaborn-white")
right_graph, right_labels = load_right()

np.random.seed(8888)
n_init = 200
clip = 1 / (right_graph.size - right_graph.shape[0])
heatmap_kws = dict(vmin=0, vmax=1, font_scale=1.5, hier_label_fontsize=20, cbar=False)

fig, ax = plt.subplots(4, 2, figsize=(15, 30))

# A priori SBM
ap_estimator = SBMEstimator()
ap_estimator.fit(right_graph, y=right_labels)

lik = ap_estimator.score(right_graph, clip=clip)

heatmap(
    right_graph,
    inner_hier_labels=right_labels,
    title="Right MB (by cell type)",
    ax=ax[0, 0],
    **heatmap_kws,
)
heatmap(
    ap_estimator.p_mat_,
    inner_hier_labels=right_labels,
    title=f"A priori SBM, lik = {lik:.2f}",
    ax=ax[0, 1],
    **heatmap_kws,
)

# A posteriori SBM
param_grid = dict(n_blocks=[4], n_components=list(range(1, 10)))
estimator = SBMEstimator
scorers = gen_scorers(estimator, right_graph)
gs = GridSweep(
    estimator,
    param_grid,
    scoring=scorers,
    n_init=n_init,
    refit="likelihood",
    verbose=5,
    n_jobs=-2,
    small_better=False,
)
gs.fit(right_graph)
pred_labels = gs.model_.vertex_assignments_
lik = gs.model_.score(right_graph, clip=clip)
heatmap(
    right_graph,
    inner_hier_labels=pred_labels,
    title="Right MB (by SBM block)",
    ax=ax[1, 0],
    **heatmap_kws,
)
heatmap(
    gs.model_.p_mat_,
    inner_hier_labels=pred_labels,
    title=f"Fit SBM, lik = {lik:.2f}",
    ax=ax[1, 1],
    **heatmap_kws,
)

# A priori DCSBM
ap_estimator = DCSBMEstimator()
ap_estimator.fit(right_graph, y=right_labels)
lik = ap_estimator.score(right_graph, clip=clip)
heatmap(
    right_graph,
    inner_hier_labels=right_labels,
    title="Right MB (by cell type)",
    ax=ax[2, 0],
    **heatmap_kws,
)
heatmap(
    ap_estimator.p_mat_,
    inner_hier_labels=right_labels,
    title=f"A priori DCSBM, lik = {lik:.2f}",
    ax=ax[2, 1],
    **heatmap_kws,
)

# A posteriori DCSBM
estimator = DCSBMEstimator
scorers = gen_scorers(estimator, right_graph)
gs = GridSweep(
    estimator,
    param_grid,
    scoring=scorers,
    n_init=n_init,
    refit="sc_likelihood",
    verbose=5,
    n_jobs=-2,
    small_better=False,
)
gs.fit(right_graph)
pred_labels = gs.model_.vertex_assignments_
lik = gs.model_.score(right_graph, clip=clip)
heatmap(
    right_graph,
    inner_hier_labels=pred_labels,
    title="Right MB (by DCSBM block)",
    ax=ax[3, 0],
    **heatmap_kws,
)
heatmap(
    gs.model_.p_mat_,
    inner_hier_labels=pred_labels,
    title=f"Fit DCSBM, lik = {lik:.2f}",
    ax=ax[3, 1],
    **heatmap_kws,
)
save("4-model", fmt="png")


#%%
from sklearn.metrics import adjusted_rand_score, confusion_matrix

adjusted_rand_score(right_labels, pred_labels)
#%%
unique_names, right_int_labels = np.unique(right_labels, return_inverse=True)

sns.set_context("talk")
plt.figure(figsize=(10, 10))
conf_mat = confusion_matrix(right_int_labels, pred_labels)
conf_mat = conf_mat[:,]
sns.heatmap(conf_mat, annot=True)


#%%

from graspy.embed import OmnibusEmbed, AdjacencySpectralEmbed
from scipy.linalg import orthogonal_procrustes

sns.set_palette("deep")
# omni = OmnibusEmbed(n_components=2)
# latent = omni.fit_transform([right_graph, gs.model_.p_mat_])
# latent = np.concatenate(latent, axis=-1)
n_components = 3
ase = AdjacencySpectralEmbed(n_components=n_components)
latent = ase.fit_transform(right_graph)
latent = np.concatenate(latent, axis=-1)

p_latent = ase.fit_transform(gs.model_.p_mat_)
p_latent = np.concatenate(p_latent, axis=-1)

R, scale = orthogonal_procrustes(p_latent, latent)
p_latent = p_latent @ R

n_components *= 2

scatter_kws = dict(legend=False, linewidth=0, s=30)
cmap1 = sns.color_palette("Set1", n_colors=4)
cmap2 = np.array(sns.color_palette("Set1", n_colors=4, desat=0.4))
cmap2 = cmap2[[3, 0, 1, 2]]
cmap2 = list(cmap2)
fig, ax = plt.subplots(n_components, n_components, figsize=(30, 30))
sns.set_context("talk")
for i in range(n_components):
    for j in range(n_components):
        a = ax[i, j]
        a.set_yticklabels([])
        a.set_xticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        if i != j:
            sns.scatterplot(
                x=latent[:, i],
                y=latent[:, j],
                hue=right_labels,
                ax=a,
                palette=cmap1,
                **scatter_kws,
            )
            sns.scatterplot(
                x=p_latent[:, i],
                y=p_latent[:, j],
                hue=pred_labels,
                ax=a,
                palette=cmap2,
                **scatter_kws,
            )
        else:
            for j, label in enumerate(np.unique(right_labels)):
                inds = np.where(right_labels == label)[0]
                plot_data = latent[inds][:, i]
                sns.distplot(plot_data, color=cmap1[j], ax=a)

plt.tight_layout()
save("multipanel_dcsbm", fmt="png")
#%%
estimator = DCSBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_dcsbm(left_adj, param_grid, n_init=25, n_jobs=-2, metric=None)

sweep_results

best_results = get_best(sweep_results, "n_params", score_name=score, small_better=False)


#
#
#
#
#
#
#
#
#
#
#
#
#
##%% #############
show_plots = False
score = "mse"

left_adj, left_labels = load_left()
if show_plots:
    heatmap(left_adj, inner_hier_labels=left_labels, cbar=False)

estimator = SBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_sbm(left_adj, param_grid, n_init=25, n_jobs=-2)

sweep_results

best_results = get_best(sweep_results, "n_params", score_name=score, small_better=False)
##%%
sns.scatterplot(data=best_results, x="n_params", y=score)
sns.scatterplot(data=ap_results, x="n_params", y=score)

##%%
estimator = DCSBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_dcsbm(left_adj, param_grid, n_init=25, n_jobs=-2, metric=None)

sweep_results

best_results = get_best(sweep_results, "n_params", score_name=score, small_better=False)
##%%
sns.scatterplot(data=best_results, x="n_params", y=score)
sns.scatterplot(data=ap_results, x="n_params", y=score)


#%%
#
