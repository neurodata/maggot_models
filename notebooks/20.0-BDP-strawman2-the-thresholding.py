#%% Load data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import ParameterGrid

from graspy.plot import gridplot, heatmap, pairplot
from src.data import load_networkx
from src.utils import meta_to_array
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


class_ids_map, class_ind_map = update_class_map(cell_ids, classes)


def update_classes(classes, pred_ids, new_class):
    new_classes = classes.copy()
    pred_inds = ids_to_inds(pred_ids)
    for i, c in zip(pred_inds, classes[pred_inds]):
        if c != "Other" and new_class not in c:
            update_name = c + "; " + new_class
        else:
            update_name = new_class
        new_classes[i] = update_name
    return new_classes


#%% Estimate the LHN neurons
# must received summed input of >= 5% from at least a SINGLE class of projection neurs
pn_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs"]
lhn_thresh = [0.05, 0.05, 0.05, 0.05]

# this is the actual search
pred_lhn_ids = proportional_search(adj, class_ind_map, pn_types, cell_ids, lhn_thresh)

# # update the predicted labels and class map
# pred_classes = update_classes(classes, pred_lhn_ids, "LHN")

# class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)

# # plot output
# if plot:
#     for t in pn_types:
#         incidence_plot(adj, pred_classes, t)


param_grid = {
    "lhn_thresh": np.linspace(0, 0.4, 50),
    "mb_thresh": np.linspace(0, 0.4, 50),
}
params = list(ParameterGrid(param_grid))

from_lhn_inds = list(class_ind_map["LHN"])
from_mb_inds = class_ind_map["MBON"]
lhn_input = adj[from_lhn_inds, :].sum(axis=0)
mb_input = adj[from_mb_inds, :].sum(axis=0)


def find_lhn2(
    adj, class_ind_map, cell_ids, pred_classes, lhn_thresh=0.05, mb_thresh=0.05
):
    mb_input_types = ["MBON"]
    lhn_input_types = ["LHN"]  # TODO should this include LHN; CN?
    # TODO the other danger here is just that for doing multiple cell types sometimes
    # would not care whether LHN or LHN; CN for the purposes of summing input
    # some cells might have not enough input from LHN or LHN; CN, but from summing both,
    # they do.

    pred_from_lhn_ids = proportional_search(
        adj, class_ind_map, lhn_input_types, cell_ids, thresh=lhn_thresh
    )
    pred_from_mb_ids = proportional_search(
        adj, class_ind_map, mb_input_types, cell_ids, thresh=mb_thresh
    )

    pred_lhn2_ids = np.setdiff1d(
        pred_from_lhn_ids, pred_from_mb_ids
    )  # get input from LHN, not MB

    pred_classes = update_classes(pred_classes, pred_lhn2_ids, "LH2N")

    # class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)
    return ids_to_inds(pred_lhn2_ids)


unknown_inds = class_ind_map["Other"]
data = np.stack((lhn_input, mb_input), axis=1)

objective_data = []


def cov_objective(class1_data, class2_data):
    try:
        class1_cov = EmpiricalCovariance().fit(class1_data).covariance_
        class2_cov = EmpiricalCovariance().fit(class2_data).covariance_
        objective = np.trace(class1_cov) + np.trace(class2_cov)
        return objective
    except ValueError:
        return np.nan


for p in params:
    test_pred_inds = run_threshold(adj, class_ind_map, cell_ids, classes, **p)
    pred_lhn_data = data[test_pred_inds, :]
    pred_unknown_inds = np.setdiff1d(unknown_inds, test_pred_inds)
    unknown_data = data[pred_unknown_inds, :]
    # print("LH2N: " + str(len(test_pred_inds)))
    # print("Other: " + str(len(pred_unknown_inds)))
    objective = cov_objective(pred_lhn_data, unknown_data)
    objective_data.append([p["lhn_thresh"], p["mb_thresh"], objective])


objective_data = np.array(objective_data)
objective_df = pd.DataFrame(
    data=objective_data, columns=["lhn_thresh", "mb_thresh", "objective"]
)
#%%
plt.figure(figsize=(15, 15))
sns.scatterplot(
    data=objective_df,
    x="lhn_thresh",
    y="mb_thresh",
    size="objective",
    hue="objective",
    sizes=(10, 600),
    legend=False,
    palette="hot",
)

#%% visualize proportion of inputs from PNs


def calculate_from_class_input(class_name, class_ind_map, adj):
    inds = class_ind_map[class_name]  # indices for neurons of that class
    from_class_adj = adj[inds, :]  # select the rows corresponding to that class
    prop_input = from_class_adj.sum(axis=0)  # sum input from that class
    return prop_input


sns.set_context("talk", font_scale=1)
pn_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs"]
pn_input_props = {}
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(15, 10))
ax = ax.ravel()
for i, t in enumerate(pn_types):
    pn_prop_input = calculate_from_class_input(t, class_ind_map, adj)
    pn_input_props[t] = pn_prop_input
    sns.distplot(pn_prop_input[pn_prop_input > 0], ax=ax[i], norm_hist=True)
    ax[i].set_title(t)


pn_prop_input_mat = np.array(list(pn_input_props.values())).T

pairplot(pn_prop_input_mat, col_names=pn_types)

#%% LHN - someone who received >5% input from at least one projection neuron type

sns.set_context("talk", font_scale=1)

max_pn_prop_input = pn_prop_input_mat.max(axis=1)
thresh_range = np.linspace(0, 0.35, num=50)


def var_objective(input, class1_inds, class2_inds):
    class1_var = np.var(input[class1_inds])
    class2_var = np.var(input[class2_inds])
    objective = class1_var + class2_var
    return objective


fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.distplot(max_pn_prop_input, ax=ax[0])
ax[0].set_title("All neurons")

objectives = np.zeros_like(thresh_range)
for i, t in enumerate(thresh_range):
    low_inds = np.where(max_pn_prop_input < t)[0]
    high_inds = np.where(max_pn_prop_input >= t)[0]
    objectives[i] = var_objective(max_pn_prop_input, low_inds, high_inds)

sns.scatterplot(x=thresh_range, y=objectives, ax=ax[1])
ax[1].set_ylim((0, 0.015))
ax[1].set_ylabel("Var objective val")
ax[1].set_xlabel("PN input threshold (min input any subclass)")
ax[1].set_xlim((0 - 0.01, 0.35 + 0.01))

## looking at only the unlabeled guys

max_pn_prop_input = max_pn_prop_input[class_ind_map["Other"]]
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.distplot(max_pn_prop_input, ax=ax[0])
ax[0].set_title("'Other' neurons")

objectives = np.zeros_like(thresh_range)
for i, t in enumerate(thresh_range):
    low_inds = np.where(max_pn_prop_input < t)[0]
    high_inds = np.where(max_pn_prop_input >= t)[0]
    objectives[i] = var_objective(max_pn_prop_input, low_inds, high_inds)

sns.scatterplot(x=thresh_range, y=objectives, ax=ax[1])
ax[1].set_ylim((0, 0.015))
ax[1].set_ylabel("Var objective val")
ax[1].set_xlabel("PN input threshold (min input any subclass)")
ax[1].set_xlim((0 - 0.01, 0.35 + 0.01))


#%%
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.utils import binarize


def dcsbm_objective(adj, labels):
    # class1_var = np.var(input[class1_inds])
    # class2_var = np.var(input[class2_inds])
    dcsbm = SBMEstimator()
    dcsbm.fit(adj, y=labels)
    objective = dcsbm.score(adj)
    return objective


fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.distplot(max_pn_prop_input, ax=ax[0])
ax[0].set_title("All neurons")

objectives = np.zeros_like(thresh_range)
for i, t in enumerate(thresh_range):
    low_inds = np.where(max_pn_prop_input < t)[0]
    high_inds = np.where(max_pn_prop_input >= t)[0]
    labels = np.zeros(adj.shape[0])
    labels[high_inds] = 1
    objectives[i] = dcsbm_objective(binarize(adj), labels)

sns.scatterplot(x=thresh_range, y=objectives, ax=ax[1])
# ax[1].set_ylim((0, 0.015))
ax[1].set_ylabel("2-DCSBM objective val")
ax[1].set_xlabel("PN input threshold (min input any subclass)")
ax[1].set_xlim((0 - 0.01, 0.35 + 0.01))

##

fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.distplot(max_pn_prop_input, ax=ax[0])
ax[0].set_title("All neurons")

objectives = np.zeros_like(thresh_range)
for i, t in enumerate(thresh_range):
    low_inds = np.where(max_pn_prop_input < t)[0]
    high_inds = np.where(max_pn_prop_input >= t)[0]
    labels = classes
    other_inds = class_ind_map["Other"]
    low_other_inds = np.intersect1d(other_inds, low_inds)
    high_other_inds = np.intersect1d(other_inds, high_inds)
    labels[low_other_inds] = "Split_low"
    labels[high_other_inds] = "Split_high"
    objectives[i] = dcsbm_objective(binarize(adj), labels)

sns.scatterplot(x=thresh_range, y=objectives, ax=ax[1])
ax[1].set_ylabel("DCSBM objective val")
ax[1].set_xlabel("PN input threshold (min input any subclass)")
ax[1].set_xlim((0 - 0.01, 0.35 + 0.01))

#%%
## looking at only the unlabeled guys

max_pn_prop_input = max_pn_prop_input[class_ind_map["Other"]]
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
sns.distplot(max_pn_prop_input, ax=ax[0])
ax[0].set_title("All neurons")

objectives = np.zeros_like(thresh_range)
for i, t in enumerate(thresh_range):
    low_inds = np.where(max_pn_prop_input < t)[0]
    high_inds = np.where(max_pn_prop_input >= t)[0]
    objectives[i] = dcsbm_objective(binarize(adj), low_inds, high_inds)

sns.scatterplot(x=thresh_range, y=objectives, ax=ax[1])
ax[1].set_ylim((0, 0.015))
ax[1].set_ylabel("Var objective val")
ax[1].set_xlabel("PN input threshold (min input any subclass)")
ax[1].set_xlim((0 - 0.01, 0.35 + 0.01))


#%%
