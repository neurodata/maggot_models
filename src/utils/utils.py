import json

import numpy as np
import pandas as pd
from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import RDPGEstimator, SBMEstimator
from graspy.simulations import p_from_latent, sample_edges, sbm


def estimate_assignments(graph, n_communities, n_components=None):
    """Given a graph and n_comunities, sweeps over covariance structures
    Not deterministic
    Not using graph bic or mse to calculate best

    1. Does an embedding on the raw graph
    2. GaussianCluster on the embedding. This will sweep covariance structure for the 
       given n_communities
    3. Returns n_parameters based on the number used in GaussianCluster

    """
    embed_graph = graph.copy()
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(
        embed_graph
    )
    if isinstance(latent, tuple):
        latent = np.concatenate(latent, axis=1)
    gc = GaussianCluster(min_components=n_communities, max_components=n_communities)
    vertex_assignments = gc.fit_predict(latent)
    return (vertex_assignments, gc.model_._n_parameters())


def hardy_weinberg(theta):
    """
    Maps a value from [0, 1] to the hardy weinberg curve.
    """
    hw = [theta ** 2, 2 * theta * (1 - theta), (1 - theta) ** 2]
    return np.array(hw).T


def estimate_sbm(graph, n_communities, n_components=None, directed=False):
    vertex_assignments, n_params = estimate_assignments(
        graph, n_communities, n_components
    )
    estimator = SBMEstimator(directed=directed, loops=False)
    estimator.fit(graph, y=vertex_assignments)
    return estimator, n_params


def estimate_rdpg(graph, n_components=None):
    estimator = RDPGEstimator(loops=False, n_components=n_components)
    estimator.fit(graph)
    if n_components is None:
        n_components = estimator.latent_.shape[0]
    n_params = graph.shape[0] * n_components
    return estimator, n_params


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


def select_sbm(graph, n_components_try_range, n_block_try_range, directed=False):
    """sweeps over n_components, n_blocks, fits an sbm for each 
    Using GaussianCluster, so will internally sweep covariance structure and pick best

    Returns n_params for the gaussian
    N_params for the sbm kinda
    rss
    score

    Maybe at some point this will sweep rank of B

    Parameters
    ----------
    graph : [type]
        [description]
    n_block_try_range : [type]
        [description]
    n_components_try_range : [type]
        [description]
    directed : bool, optional
        [description], by default False
    """
    n_exps = len(n_components_try_range) * len(n_block_try_range)
    columns = [
        "n_params_gmm",
        "n_params_sbm",
        "rss",
        "mse",
        "score",
        "n_components_try",
        "n_block_try",
    ]
    out_df = pd.DataFrame(np.nan, index=range(n_exps), columns=columns)

    for i, n_components_try in enumerate(n_components_try_range):
        for j, n_block_try in enumerate(n_block_try_range):
            # TODO figure out low rank B?
            # TODO try tommy clust instead here
            estimator, n_params_gmm = estimate_sbm(
                graph, n_block_try, n_components=n_components_try, directed=False
            )
            rss = compute_rss(estimator, graph)
            mse = compute_mse(estimator, graph)
            score = compute_log_lik(estimator, graph)
            n_params_sbm = estimator._n_parameters()

            ind = i * len(n_block_try_range) + j
            out_df.loc[ind, "n_params_gmm"] = n_params_gmm
            out_df.loc[ind, "n_params_sbm"] = n_params_sbm
            out_df.loc[ind, "rss"] = rss
            out_df.loc[ind, "mse"] = mse
            out_df.loc[ind, "score"] = score
            out_df.loc[ind, "n_components_try"] = n_components_try
            out_df.loc[ind, "n_block_try"] = n_block_try

    return out_df


def run_to_df(file_path):
    f = open(str(file_path), mode="r")
    out = json.load(f)
    f.close()
    data_dict = out["result"]["values"]
    return pd.DataFrame.from_dict(data_dict)
