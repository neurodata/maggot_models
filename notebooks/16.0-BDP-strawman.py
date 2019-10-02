#%% Load data
import numpy as np
import networkx as nx
from graspy.plot import heatmap, gridplot
from src.data import load_networkx
from src.utils import meta_to_array
from src.visualization import incidence_plot

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

    pred_cell_ids = []
    for i, class_name in enumerate(or_classes):
        inds = class_ind_map[class_name]  # indices for neurons of that class
        from_class_adj = adj[inds, :]  # select the rows corresponding to that class
        prop_input = from_class_adj.sum(axis=0)  # sum input from that class
        # prop_input /= adj.sum(axis=0)
        if thresh[i] > 0:
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
for t in pn_types:
    incidence_plot(adj, pred_classes, t)

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

for t in mb_input_types + innate_input_types:
    incidence_plot(adj, pred_classes, t)

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

for t in mb_input_types + innate_input_types:
    incidence_plot(adj, pred_classes, t)
# #%%
# inds = ids_to_inds(pred_mb2_ids)
# adj[:, inds][class_ind_map["MBON"], :].sum(axis=0)


# #%%
# adj[:, inds][class_ind_map["LHN"], :].sum(axis=0)

#%% Estimate Second-order LHN
mb_input_types = ["MBON"]
mb_thresh = [0.05]

lhn_input_types = ["LHN"]  # TODO should this include LHN; CN?
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

pred_classes = update_classes(pred_classes, pred_lhn2_ids, "LH2N")

class_ids_map, class_ind_map = update_class_map(cell_ids, pred_classes)

for t in mb_input_types + innate_input_types:
    incidence_plot(adj, pred_classes, t)
#%%#
inds = ids_to_inds(pred_lhn2_ids)

adj[class_ind_map["LHN"], :][:, class_ind_map["LH2N"]].sum(axis=0)


#%%
