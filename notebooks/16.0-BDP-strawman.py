#%% Load data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from graspy.plot import gridplot, heatmap
from src.data import load_networkx
from src.utils import meta_to_array, savefig
from src.visualization import incidence_plot

plt.style.use("seaborn-white")
sns.set_palette("deep")
plot = True

PREDEFINED_CLASSES = [
    "DANs",
    "KCs",
    "MBINs",
    "MBON",
    "MBON; CN",
    "OANs",
    "ORN mPNs",
    "ORN uPNs",
    "tPNs",
    "vPNs",
]

graph_type = "Gn"

graph = load_networkx(graph_type)

heatmap(graph, transform="simple-all")

df_adj = nx.to_pandas_adjacency(graph)
adj = df_adj.values

classes = meta_to_array(graph, "Class")
classes = classes.astype("<U64")
classes[classes == "MBON; CN"] = "MBON"
print(np.unique(classes))


nx_ids = np.array(list(graph.nodes()), dtype=int)
df_ids = df_adj.index.values.astype(int)
print("nx indexed same as pd")
print(np.array_equal(nx_ids, df_ids))
cell_ids = df_ids

PREDEFINED_IDS = []
for c, id in zip(classes, cell_ids):
    if c in PREDEFINED_CLASSES:
        PREDEFINED_IDS.append(id)

IDS_TO_INDS_MAP = dict(zip(cell_ids, range(len(cell_ids))))


def ids_to_inds(ids):
    inds = [IDS_TO_INDS_MAP[k] for k in ids]
    return inds


def proportional_search(adj, class_ind_map, or_classes, ids, thresh):
    """finds the cell ids of neurons who receive a certain proportion of their 
    input from one of the cells in or_classes 
    
    Parameters
    ----------
    adj : np.array
        adjacency matrix, assumed to be normalized so that columns sum to 1
    class_map : dict
        keys are class names, values are arrays of indices describing where that class
        can be found in the adjacency matrix
    or_classes : list 
        which classes to consider for the input thresholding. Neurons will be selected 
        which satisfy ANY of the input threshold criteria
    ids : np.array
        names of each cell 
    """
    if not isinstance(thresh, list):
        thresh = [thresh]

    pred_cell_ids = []
    for i, class_name in enumerate(or_classes):
        inds = class_ind_map[class_name]  # indices for neurons of that class
        from_class_adj = adj[inds, :]  # select the rows corresponding to that class
        prop_input = from_class_adj.sum(axis=0)  # sum input from that class
        # prop_input /= adj.sum(axis=0)
        if thresh[i] >= 0:
            flag_inds = np.where(prop_input >= thresh[i])[0]  # inds above threshold
        elif thresh[i] < 0:
            flag_inds = np.where(prop_input <= -thresh[i])[0]  # inds below threshold
        pred_cell_ids += list(ids[flag_inds])  # append to cells which satisfied

    pred_cell_ids = np.unique(pred_cell_ids)

    pred_cell_ids = np.setdiff1d(pred_cell_ids, PREDEFINED_IDS)

    return pred_cell_ids


#%% Map MW classes to the indices of cells belonging to them
def update_class_map(cell_ids, classes):
    unique_classes, inverse_classes = np.unique(classes, return_inverse=True)
    class_ind_map = {}
    class_ids_map = {}
    for i, class_name in enumerate(unique_classes):
        inds = np.where(inverse_classes == i)[0]
        ids = cell_ids[inds]
        class_ind_map[class_name] = inds
        class_ids_map[class_name] = ids
    return class_ids_map, class_ind_map


og_class_ids_map, og_class_ind_map = update_class_map(cell_ids, classes)


def update_classes(classes, pred_ids, new_class, mode="append"):
    new_classes = classes.copy()
    pred_inds = ids_to_inds(pred_ids)
    for i, c in zip(pred_inds, classes[pred_inds]):
        if mode == "append":
            if c != "Other" and new_class not in c:
                update_name = c + "; " + new_class
            else:
                update_name = new_class
        elif mode == "replace":
            update_name = new_class
        new_classes[i] = update_name
    return new_classes


#%% Estimate the LHN neurons
# must received summed input of >= 5% from at least a SINGLE class of projection neurs
pn_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs"]
lhn_thresh = [0.05, 0.05, 0.05, 0.05]

# this is the actual search
pred_lhn_ids = proportional_search(
    adj, og_class_ind_map, pn_types, cell_ids, lhn_thresh
)

# this is just for estimating how well I did relative to Michael
true_lhn_inds = np.concatenate((og_class_ind_map["LHN"], og_class_ind_map["LHN; CN"]))
true_lhn_ids = df_ids[true_lhn_inds]

print("LHN")
print("Recall:")
print(np.isin(true_lhn_ids, pred_lhn_ids).mean())  # how many of the og lhn i got
print("Precision:")
print(np.isin(pred_lhn_ids, true_lhn_ids).mean())  # this is how many of mine are in og
print(len(pred_lhn_ids))

# update the predicted labels and class map
pred_classes = update_classes(classes, pred_lhn_ids, "LHN")

class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)

# plot output
if plot:
    for t in pn_types:
        incidence_plot(adj, pred_classes, t)
        savefig("lhn_" + t)
#%% Estimate CN neurons
innate_input_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs", "LHN"]
innate_thresh = 5 * [0.05]

mb_input_types = ["MBON"]
mb_thresh = [0.05]

pred_innate_ids = proportional_search(
    adj, class_ind_map, innate_input_types, cell_ids, thresh=innate_thresh
)
pred_learn_ids = proportional_search(
    adj, class_ind_map, mb_input_types, cell_ids, thresh=mb_thresh
)
pred_cn_ids = np.intersect1d(pred_learn_ids, pred_innate_ids)  # get input from both

true_cn_ids = (og_class_ids_map["CN"], og_class_ids_map["LHN; CN"])
true_cn_ids = np.concatenate(true_cn_ids)

print("CN")
print("Recall:")
print(np.isin(true_cn_ids, pred_cn_ids).mean())  # how many of the og lhn i got
print("Precision:")
print(np.isin(pred_cn_ids, true_cn_ids).mean())  # this is how many of mine are in og
print(len(pred_cn_ids))

pred_classes = update_classes(pred_classes, pred_cn_ids, "CN")

class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)

if plot:
    for t in mb_input_types + innate_input_types:
        incidence_plot(adj, pred_classes, t)
        savefig("cn_" + t)
#%% Estimate Second-order Mushroom Body
mb_input_types = ["MBON"]
mb_thresh = [0.05]

innate_input_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs", "LHN"]
innate_thresh = 5 * [0.05]

pred_learn_ids = proportional_search(
    adj, class_ind_map, mb_input_types, cell_ids, thresh=mb_thresh
)
pred_innate_ids = proportional_search(
    adj, class_ind_map, innate_input_types, cell_ids, thresh=innate_thresh
)

pred_mb2_ids = np.setdiff1d(
    pred_learn_ids, pred_innate_ids
)  # get input from mb, not innate

pred_classes = update_classes(pred_classes, pred_mb2_ids, "MB20N")

class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)

if plot:
    for t in mb_input_types + innate_input_types:
        incidence_plot(adj, pred_classes, t)
        savefig("mb2on_" + t)

#%%
print("Sum of input from MBON to MB2ON")
inds = ids_to_inds(pred_mb2_ids)
print(adj[:, inds][class_ind_map["MBON"], :].sum(axis=0))
print("Min")
print(adj[:, inds][class_ind_map["MBON"], :].sum(axis=0).min())
print()
print("Sum of input from LHN to MB20N")
inds = ids_to_inds(pred_mb2_ids)
print(adj[:, inds][class_ind_map["LHN"], :].sum(axis=0))
print("Max")
print(adj[:, inds][class_ind_map["LHN"], :].sum(axis=0).max())
print()

#%% Estimate Second-order LHN
mb_input_types = ["MBON"]
mb_thresh = [0.05]

lhn_input_types = ["LHN"]  # TODO should this include LHN; CN?
# TODO the other danger here is just that for doing multiple cell types sometimes you
# would not care whether LHN or LHN; CN for the purposes of summing input
# some cells might have not enough input from LHN or LHN; CN, but from summing both,
# they do.
lhn_thresh = [0.05]

pred_from_lhn_ids = proportional_search(
    adj, class_ind_map, lhn_input_types, cell_ids, thresh=lhn_thresh
)
pred_from_mb_ids = proportional_search(
    adj, class_ind_map, mb_input_types, cell_ids, thresh=mb_thresh
)

pred_lhn2_ids = np.setdiff1d(
    pred_from_lhn_ids, pred_from_mb_ids
)  # get input from LHN, not MB

pred_classes = update_classes(pred_classes, pred_lhn2_ids, "LH2N", mode="replace")

class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)

#%% save output
old_meta_df_path = "maggot_models/data/raw/Maggot-Brain-Connectome/"
old_meta_df_path += "4-color-matrices_Brain/2019-09-18-v2/brain_meta-data.csv"


old_meta_df = pd.read_csv(old_meta_df_path, index_col=0)
print(old_meta_df.head())
#%%
classes = []
ids = []
for key, vals in class_ids_map.items():
    class_indicator = len(vals) * [key]
    classes += class_indicator
    ids += list(vals)

meta_df = pd.DataFrame()
meta_df["Class"] = classes
meta_df.index = ids
new_meta_df = meta_df.loc[old_meta_df.index]
new_meta_df.head()
meta_df = old_meta_df.copy()
meta_df["BP_Class"] = new_meta_df["Class"].values
print(meta_df.head())

out_loc = "maggot_models/data/processed/2019-09-18-v2/BP_metadata.csv"
meta_df.to_csv(out_loc)
# lhn_input_types = [["CN; LHN; LH2N", "LHN", "LHN; CN", "LHN; LH2N"]]
# lhn_input_types = [["LHN", "LHN; LH2N"]]  # these are the ones we chose on
#%%
if plot:
    for t in mb_input_types + lhn_input_types:
        incidence_plot(adj, pred_classes, t)
        if "LHN" in t:
            savefig("lh2n_lhn")
        else:
            savefig("lh2n_" + str(t))
#%%#
print(np.unique(pred_classes))

heatmap(
    adj,
    inner_hier_labels=pred_classes,
    figsize=(20, 20),
    sort_nodes=True,
    hier_label_fontsize=10,
    transform="simple-all",
)
savefig("updated_heatmap", fmt="png")

# #%% try a plot of MBON input vs LHN

# # set up data
# from_lhn_inds = list(class_ind_map["LHN"]) + list(class_ind_map["LHN; LH2N"])
# from_mb_inds = class_ind_map["MBON"]
# lhn_input = adj[from_lhn_inds, :].sum(axis=0)
# mb_input = adj[from_mb_inds, :].sum(axis=0)

# class_types = ["Hard class", "LH2N; *", "Not"]
# plot_classes = []
# for id, class_name in zip(cell_ids, pred_classes):
#     if class_name in PREDEFINED_CLASSES:
#         plot_classes.append("Hard class")
#     elif "LH2N" in class_name:
#         plot_classes.append("LH2N; *")
#     else:
#         plot_classes.append("Not")

# plot_df = pd.DataFrame(columns=["lhn_input", "mb_input", "class"])
# plot_df["lhn_input"] = lhn_input
# plot_df["mb_input"] = mb_input
# plot_df["class"] = plot_classes

# sns.set_context("talk", font_scale=1.5)
# sns.set_palette("Set1")

# # scatterplot
# plt.figure(figsize=(15, 15))
# ax = sns.scatterplot(
#     data=plot_df,
#     x="lhn_input",
#     y="mb_input",
#     hue="class",
#     hue_order=class_types,
#     alpha=0.3,
#     s=20,
# )

# # add lines for the boundaries
# ax.axvline(0.05, c="k", linestyle="--", alpha=0.5)
# ax.axhline(0.05, c="k", linestyle="--", alpha=0.5)

# # add marginals
# divider = make_axes_locatable(ax)

# # right marginal
# ax_right = divider.new_horizontal(size="15%", pad=0.05, pack_start=False, sharey=ax)
# ax.figure.add_axes(ax_right)
# ax_right.axis("off")
# bins = np.arange(0, 0.6, 0.02)
# for class_name in class_types:
#     data = plot_df[plot_df["class"] == class_name]
#     x = data["mb_input"].values
#     sns.distplot(
#         x,
#         ax=ax_right,
#         vertical=True,
#         kde=False,
#         bins=bins,
#         norm_hist=True,
#         # hist_kws=hist_kws,
#     )

# # left marginal
# ax_top = divider.new_vertical(size="15%", pad=0.05, pack_start=False, sharex=ax)
# ax.figure.add_axes(ax_top)
# ax_top.axis("off")
# bins = np.arange(0, 0.8, 0.02)
# for class_name in class_types:
#     data = plot_df[plot_df["class"] == class_name]
#     x = data["lhn_input"].values
#     sns.distplot(x, ax=ax_top, vertical=False, kde=False, bins=bins, norm_hist=True)

# savefig("lh2n_partition")
# #%% Try automated splitting
# from sklearn.model_selection import ParameterGrid
# from sklearn.covariance import EmpiricalCovariance

# param_grid = {
#     "lhn_thresh": np.linspace(0, 0.4, 50),
#     "mb_thresh": np.linspace(0, 0.4, 50),
# }
# params = list(ParameterGrid(param_grid))

# from_lhn_inds = list(class_ind_map["LHN"])
# from_mb_inds = class_ind_map["MBON"]
# lhn_input = adj[from_lhn_inds, :].sum(axis=0)
# mb_input = adj[from_mb_inds, :].sum(axis=0)


# def find_lhn2(
#     adj, class_ind_map, cell_ids, pred_classes, lhn_thresh=0.05, mb_thresh=0.05
# ):
#     mb_input_types = ["MBON"]
#     lhn_input_types = ["LHN"]  # TODO should this include LHN; CN?
#     # TODO the other danger here is just that for doing multiple cell types sometimes
#     # would not care whether LHN or LHN; CN for the purposes of summing input
#     # some cells might have not enough input from LHN or LHN; CN, but from summing both,
#     # they do.

#     pred_from_lhn_ids = proportional_search(
#         adj, class_ind_map, lhn_input_types, cell_ids, thresh=lhn_thresh
#     )
#     pred_from_mb_ids = proportional_search(
#         adj, class_ind_map, mb_input_types, cell_ids, thresh=mb_thresh
#     )

#     pred_lhn2_ids = np.setdiff1d(
#         pred_from_lhn_ids, pred_from_mb_ids
#     )  # get input from LHN, not MB

#     pred_classes = update_classes(pred_classes, pred_lhn2_ids, "LH2N",)

#     # class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)
#     return ids_to_inds(pred_lhn2_ids)


# unknown_inds = class_ind_map["Other"]
# data = np.stack((lhn_input, mb_input), axis=1)

# objective_data = []


# def cov_objective(class1_data, class2_data):
#     try:
#         class1_cov = EmpiricalCovariance().fit(class1_data).covariance_
#         class2_cov = EmpiricalCovariance().fit(class2_data).covariance_
#         objective = np.trace(class1_cov) + np.trace(class2_cov)
#         return objective
#     except ValueError:
#         return np.nan


# for p in params:
#     test_pred_inds = find_lhn2(adj, class_ind_map, cell_ids, pred_classes, **p)
#     pred_lhn_data = data[test_pred_inds, :]
#     pred_unknown_inds = np.setdiff1d(unknown_inds, test_pred_inds)
#     unknown_data = data[pred_unknown_inds, :]
#     # print("LH2N: " + str(len(test_pred_inds)))
#     # print("Other: " + str(len(pred_unknown_inds)))
#     objective = cov_objective(pred_lhn_data, unknown_data)
#     objective_data.append([p["lhn_thresh"], p["mb_thresh"], objective])


# objective_data = np.array(objective_data)
# objective_df = pd.DataFrame(
#     data=objective_data, columns=["lhn_thresh", "mb_thresh", "objective"]
# )
# #%%
# plt.figure(figsize=(15, 15))
# sns.scatterplot(
#     data=objective_df,
#     x="lhn_thresh",
#     y="mb_thresh",
#     size="objective",
#     hue="objective",
#     sizes=(10, 600),
#     legend=False,
#     palette="hot",
# )
# #%%

# #%% ###

# # inds = ids_to_inds(pred_lhn2_ids)
# # from_inds = list(class_ind_map["LHN"]) + list(class_ind_map["LHN; LH2N"])
# # adj[from_inds, :][:, inds].sum(axis=0)

# # plt.figure(figsize=(15, 15))

# # data = plot_df[plot_df["class"] == "Not"]
# # sns.jointplot(
# #     data=data, x="lhn_input", y="mb_input", alpha=0.5, size=10, s=2, kind="kde"
# # )
# # data = plot_df[plot_df["class"] == "LH2N; *"]
# # ax2 = sns.jointplot(
# #     data=data, x="lhn_input", y="mb_input", alpha=0.5, size=10, s=2, kind="kde"
# # )


# # plt.figure(figsize=(15, 15))
# # # data = plot_df[plot_df["class"] == "Not"]
# # # x = data["lhn_input"]
# # # y = data["mb_input"]
# # # ax1 = sns.kdeplot(
# # #     data=x, data2=y, alpha=0.5, size=10, s=2, kind="kde", cmap="Blues", n_levels=30
# # # )
# # data = plot_df[plot_df["class"] == "LH2N; *"]
# # x = data["lhn_input"]
# # y = data["mb_input"]
# # ax2 = sns.kdeplot(
# #     data=x, data2=y, alpha=0.5, size=10, s=2, kind="kde", cmap="Reds", n_levels=30
# # )

# # divider = make_axes_locatable(ax2)
# # ax_right = divider.new_horizontal(size="10%", pad=0.0, pack_start=False)
# # ax.figure.add_axes(ax_right)
# # sns.distplot(y, ax=ax_right)


# #%%
