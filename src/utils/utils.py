import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from graspy.simulations import p_from_latent, sample_edges, sbm

from graspy.models import SBMEstimator


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
