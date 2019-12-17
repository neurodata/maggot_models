import json
import os
import pickle
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.models import SBMEstimator
from graspy.simulations import p_from_latent, sample_edges, sbm


def hardy_weinberg(theta):
    """
    Maps a value from [0, 1] to the hardy weinberg curve.
    """
    hw = [theta ** 2, 2 * theta * (1 - theta), (1 - theta) ** 2]
    return np.array(hw).T


def gen_hw_graph(n_verts):
    thetas = np.random.uniform(0, 1, n_verts)
    latent = hardy_weinberg(thetas)
    p_mat = p_from_latent(latent, rescale=False, loops=False)
    graph = sample_edges(p_mat, directed=True, loops=False)
    return (graph, p_mat)


def compute_rss(estimator, graph):
    """Computes RSS, matters whether the estimator is directed
    
    Parameters
    ----------
    estimator : graspy estimator object
        [description]
    graph : nparray
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    graph = graph.copy()
    p_mat = estimator.p_mat_.copy()
    if not estimator.directed:
        inds = np.triu_indices_from(p_mat)
        p_mat = p_mat[inds]
        graph = graph[inds]
    diff = (p_mat - graph) ** 2
    rss = np.sum(diff)
    return rss


def compute_mse(estimator, graph):
    """
    Matters whether the estimator is directed
    """
    rss = compute_rss(estimator, graph)
    if not estimator.directed:  # TODO double check that this is right
        size = graph.shape[0] * (graph.shape[0] - 1) / 2
    else:
        size = graph.size - graph.shape[0]
    return rss / size


def compute_log_lik(estimator, graph, c=0):
    """This is probably wrong right now"""
    p_mat = estimator.p_mat_.copy()
    graph = graph.copy()
    inds = np.triu_indices(graph.shape[0])
    p_mat = p_mat[inds]
    graph = graph[inds]

    p_mat[p_mat < c] = c
    p_mat[p_mat > 1 - c] = 1 - c
    successes = np.multiply(p_mat, graph)
    failures = np.multiply((1 - p_mat), (1 - graph))
    likelihood = successes + failures
    return np.sum(np.log(likelihood))


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def gen_B(n_blocks, a=0.1, b=0.2, assortivity=4):
    B_mat = np.random.uniform(a, b, size=(n_blocks, n_blocks))
    B_mat -= np.diag(np.diag(B_mat))
    B_mat += np.diag(np.random.uniform(assortivity * a, assortivity * b, size=n_blocks))
    return B_mat


def gen_sbm(n_verts, n_blocks, B_mat):
    ps = np.array(n_blocks * [1 / n_blocks])
    n_vec = np.random.multinomial(n_verts, ps)
    graph = sbm(n_vec, B_mat, directed=False, loops=False)
    labels = _n_to_labels(n_vec)
    return graph, labels


def run_to_df(file_path):
    out = get_json(file_path)
    result = out["result"]
    if "py/tuple" in result:
        dfs = []
        for elem in result["py/tuple"]:
            df = pd.DataFrame.from_dict(elem["values"])
            dfs.append(df)
        return dfs
    else:
        print(result["values"][:100])
        return pd.DataFrame.from_dict(result["values"])


def get_json(file_path):
    f = open(str(file_path), mode="r")
    out = json.load(f)
    f.close()
    return out


def compute_mse_from_assignments(assignments, graph, directed=True, loops=False):
    estimator = SBMEstimator(loops=loops, directed=directed)
    estimator.fit(graph, y=assignments)
    return compute_mse(estimator, graph)


def get_best_df(input_df):
    """super hard coded right now (e.g. column names)
    
    Parameters
    ----------
    df : dataframe
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    param_df = input_df[input_df["sim_ind"] == 0]
    labels = ["n_block_try", "n_components_try", "mse"]
    param_df = param_df.loc[:, labels]
    param_df["best_sim"] = 0
    param_df["best_ind"] = 0
    for i in range(50):
        df = input_df[input_df["sim_ind"] == i]
        for j, row in df.iterrows():
            temp_df = param_df.loc[
                (param_df[labels[0]] == row[labels[0]])
                & (param_df[labels[1]] == row[labels[1]])
            ]
            ind = temp_df.index
            if row["mse"] <= param_df.loc[ind, "mse"].values[0]:
                param_df.loc[ind, "mse"] = row["mse"]
                param_df.loc[ind, "best_sim"] = row["sim_ind"]
                param_df.loc[ind, "best_ind"] = j
    best_df = input_df.loc[param_df["best_ind"].values, :]
    return best_df


def get_best_df2(input_df):
    """super hard coded right now (e.g. column names)
    
    Parameters
    ----------
    df : dataframe
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    # param_df = input_df[input_df["sim_ind"] == 0]
    labels = ["n_block_try", "mse"]
    param_df = pd.DataFrame()
    # param_df = param_df.loc[:, labels]
    param_df["n_block_try"] = np.unique(input_df["n_block_try"].values)
    param_df["best_sim"] = 0
    param_df["best_ind"] = 0
    param_df["mse"] = np.inf
    for i in range(max(input_df["sim_ind"].values) + 1):
        df = input_df[input_df["sim_ind"] == i]
        for j, row in df.iterrows():
            temp_df = param_df.loc[(param_df[labels[0]] == row[labels[0]])]
            ind = temp_df.index
            if row["mse"] <= param_df.loc[ind, "mse"].values[0]:
                param_df.loc[ind, "mse"] = row["mse"]
                param_df.loc[ind, "best_sim"] = row["sim_ind"]
                param_df.loc[ind, "best_ind"] = j
    best_df = input_df.loc[param_df["best_ind"].values, :]
    return best_df


def get_best_df3(input_df):
    """super hard coded right now (e.g. column names)
    
    Parameters
    ----------
    df : dataframe
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    param_df = input_df[input_df["sim_ind"] == 0]
    labels = ["n_block_try", "rank_try", "mse"]
    param_df = param_df.loc[:, labels]
    param_df["best_sim"] = 0
    param_df["best_ind"] = 0
    for i in range(50):
        df = input_df[input_df["sim_ind"] == i]
        for j, row in df.iterrows():
            temp_df = param_df.loc[
                (param_df[labels[0]] == row[labels[0]])
                & (param_df[labels[1]] == row[labels[1]])
            ]
            ind = temp_df.index
            if row["mse"] <= param_df.loc[ind, "mse"].values[0]:
                param_df.loc[ind, "mse"] = row["mse"]
                param_df.loc[ind, "best_sim"] = row["sim_ind"]
                param_df.loc[ind, "best_ind"] = j
    best_df = input_df.loc[param_df["best_ind"].values, :]
    return best_df


def load_config(path, experiment, run):
    exp_path = Path(path)
    exp_path = exp_path / experiment
    exp_path = exp_path / str(run)
    run_path = exp_path / "run.json"
    config_path = exp_path / "config.json"

    config = get_json(config_path)
    print(f"Experiment: {experiment}")
    print(f"Run: {run}")
    print(f"Path: {run_path}")
    print()
    print("Experiment configuration:")
    print()
    for key, value in config.items():
        print(key)
        print(value)
    return config


def load_run(path, experiment, run):
    exp_path = Path(path)
    exp_path = exp_path / experiment
    exp_path = exp_path / str(run)
    run_path = exp_path / "run.json"

    try:
        dfs = run_to_df(run_path)
        return dfs
    except:
        print("Could not find df in run")


def load_pickle(path, experiment, run, name="master_out_df"):
    exp_path = Path(path)
    exp_path = exp_path / experiment
    exp_path = exp_path / str(run)
    exp_path = exp_path / str(name + ".pickle")
    with open(exp_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_obj(obj, fso, name):
    path = fso.dir
    path = Path(path)
    path = path / str(name + ".pickle")
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    print(f"Saved to {path}")


def get_best(df, param_name="param_n_components", score_name="mse", small_better=True):
    param_range = np.unique(df[param_name].values)
    best_rows = []
    for param_value in param_range:
        temp_df = df[df[param_name] == param_value]
        if small_better:
            ind = temp_df[score_name].idxmin()  # this is the metric we are choosing on
        else:
            ind = temp_df[score_name].idxmax()
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


def get_subgraph(graph, feature, key):
    """return the subgraph of a networkx object

    based on the node data "feature" being equal to "key"
    
    Parameters
    ----------
    graph : [type]
        [description]
    feature : [type]
        [description]
    key : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    sub_nodes = [node for node, meta in graph.nodes(data=True) if meta[feature] == key]
    return graph.subgraph(sub_nodes)


def to_simple_class(classes):
    if not isinstance(classes, (list, np.ndarray)):
        classes = [classes]
    name_map = {
        "CN": "C/LH",
        "DANs": "I",
        "KCs": "KC",
        "LHN": "C/LH",
        "LHN; CN": "C/LH",
        "MBINs": "I",
        "MBON": "O",
        "MBON; CN": "O",
        "OANs": "I",
        "ORN mPNs": "P",
        "ORN uPNs": "P",
        "tPNs": "P",
        "vPNs": "P",
        "Unidentified": "U",
        "Other": "U",
    }
    simple_classes = np.array(itemgetter(*classes)(name_map))
    return simple_classes


def meta_to_array(graph, key, nodelist=None):

    # if nodelist is not None:
    #     nodelist_map = dict(zip(nodelist, range(len(nodelist))))

    # data = np.zeros(len(graph), dtype=)
    # for node, meta in graph.nodes(data=True):
    #     node_ind = nodelist_map[node]
    #     data[node_ind] = meta

    data = [meta[key] for node, meta in graph.nodes(data=True)]

    return np.array(data)


def get_simple(graph):
    classes = meta_to_array(graph, "Class")
    simple_classes = to_simple_class(classes)
    return simple_classes


def savefig(
    name,
    fmt="pdf",
    foldername=None,
    subfoldername="figs",
    pathname="./maggot_models/notebooks/outs",
    bbox_inches="tight",
    pad_inches=0.5,
    **kws,
):
    path = Path(pathname)
    if foldername is not None:
        path = path / foldername
        if not os.path.isdir(path):
            os.mkdir(path)
        if subfoldername is not None:
            path = path / subfoldername
            if not os.path.isdir(path):
                os.mkdir(path)
    plt.savefig(
        path / str(name + "." + fmt),
        fmt=fmt,
        facecolor="w",
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        **kws,
    )


def relabel(labels):
    """
    Remaps integer labels based on who is most frequent
    """
    uni_labels, uni_inv, uni_counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    sort_inds = np.argsort(uni_counts)[::-1]
    new_labels = range(len(uni_labels))
    uni_labels_sorted = uni_labels[sort_inds]
    relabel_map = dict(zip(uni_labels_sorted, new_labels))

    new_labels = np.array(itemgetter(*labels)(relabel_map))
    return new_labels


def unique_by_size(data):
    """Equivalent to np.unique but returns data in order sorted by frequency of values
    
    Parameters
    ----------
    data : np.ndarray
        array on which to find unique values
    
    Returns
    -------
    np.ndarray
        unique elements in `data` sorted by frequency, with the most observations first
    
    np.ndarray
        counts of the unique elements in `data`
    """
    unique_data, counts = np.unique(data, return_counts=True)
    sort_inds = np.argsort(counts)[::-1]  # reverse order to get largest class first
    unique_data = unique_data[sort_inds]
    counts = counts[sort_inds]
    return unique_data, counts


def export_skeleton_json(
    name,
    ids,
    colors,
    palette="tab10",
    foldername=None,
    subfoldername="jsons",
    pathname="./maggot_models/notebooks/outs",
    multiout=False,
):
    """ Take a list of skeleton ids and output as json file for catmaid
    
    Parameters
    ----------
    name : str
        filename to save output
    ids : list or array
        skeleton ids
    colors : list or array
        either a hexadecimal color for each skeleton or a label for each skeleton to be 
        colored by palette
    palette : str or None, optional
        if not None, this is a palette specification to use to color skeletons
    """
    og_colors = colors.copy()
    uni_labels = np.unique(colors)
    n_labels = len(uni_labels)

    if palette is not None:
        pal = sns.color_palette(palette, n_colors=n_labels)
        pal = pal.as_hex()
        uni_labels = [int(i) for i in uni_labels]
        colormap = dict(zip(uni_labels, pal))
        colors = np.array(itemgetter(*colors)(colormap))

    opacs = np.array(len(ids) * [1])

    path = Path(pathname)
    if foldername is not None:
        path = path / foldername
        if not os.path.isdir(path):
            os.mkdir(path)
        if subfoldername is not None:
            path = path / subfoldername
            if not os.path.isdir(path):
                os.mkdir(path)

    if multiout:
        for l in uni_labels:
            filename = path / str(name + "_" + str(l) + ".json")

            inds = np.where(og_colors == l)[0]

            spec_list = [
                {"skeleton_id": int(i), "color": str(c), "opacity": float(o)}
                for i, c, o in zip(ids[inds], colors[inds], opacs[inds])
            ]
            with open(filename, "w") as fout:
                json.dump(spec_list, fout)
    else:
        spec_list = [
            {"skeleton_id": int(i), "color": str(c), "opacity": float(o)}
            for i, c, o in zip(ids, colors, opacs)
        ]
        filename = path / str(name + ".json")
        with open(filename, "w") as fout:
            json.dump(spec_list, fout)

    if palette is not None:
        return (spec_list, colormap, pal)
    else:
        return spec_list


def shuffle_edges(A):
    n_verts = A.shape[0]
    A_fake = A.copy().ravel()
    np.random.shuffle(A_fake)
    A_fake = A_fake.reshape((n_verts, n_verts))
    return A_fake
