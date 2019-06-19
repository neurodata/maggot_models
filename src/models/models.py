import numpy as np
import pandas as pd

from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import RDPGEstimator, SBMEstimator, EREstimator

from graspy.utils import is_symmetric

from ..utils import compute_rss, compute_mse
from .brute_cluster import brute_cluster


def estimate_assignments(graph, n_communities, n_components=None, method="gc"):
    """Given a graph and n_comunities, sweeps over covariance structures
    Not deterministic
    Not using graph bic or mse to calculate best

    1. Does an embedding on the raw graph
    2. GaussianCluster on the embedding. This will sweep covariance structure for the 
       given n_communities
    3. Returns n_parameters based on the number used in GaussianCluster

    method can be "gc" or "bc" 
    """
    embed_graph = graph.copy()
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(
        embed_graph
    )
    if isinstance(latent, tuple):
        latent = np.concatenate(latent, axis=1)
    if method == "gc":
        gc = GaussianCluster(min_components=n_communities, max_components=n_communities)
        vertex_assignments = gc.fit_predict(latent)
        n_params = gc.model_._n_parameters()
    elif method == "bc":
        vertex_assignments, n_params = brute_cluster(
            latent, [n_communities], covariance_types=["full"]
        )
    elif method == "bc-none":
        vertex_assignments, n_params = brute_cluster(
            latent,
            [n_communities],
            affinities=["none"],
            linkages=["none"],
            covariance_types=["full"],
        )
    else:
        raise ValueError("Unspecified clustering method")
    return (vertex_assignments, n_params)


def estimate_sbm(graph, n_communities, n_components=None, directed=False, method="gc"):
    if n_communities == 1:
        estimator = EREstimator(directed=directed, loops=False)
        estimator.fit(graph)
        n_params = estimator._n_parameters()
    else:
        vertex_assignments, n_params = estimate_assignments(
            graph, n_communities, n_components, method=method
        )
        estimator = SBMEstimator(directed=directed, loops=False)
        estimator.fit(graph, y=vertex_assignments)
    return estimator, n_params


def estimate_rdpg(graph, n_components=None):
    estimator = RDPGEstimator(loops=False, n_components=n_components)
    estimator.fit(graph)
    if n_components is None:
        n_components = estimator.latent_.shape[0]
    # n_params = graph.shape[0] * n_components
    n_params = estimator._n_parameters()
    return estimator, n_params


def select_sbm(
    graph, n_components_try_range, n_block_try_range, directed=False, method="gc"
):
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
    c = 1 / (graph.size - graph.shape[0])

    for i, n_components_try in enumerate(n_components_try_range):
        for j, n_block_try in enumerate(n_block_try_range):
            # TODO figure out low rank B?
            # TODO try tommy clust instead here
            estimator, n_params_gmm = estimate_sbm(
                graph,
                n_block_try,
                n_components=n_components_try,
                directed=False,
                method=method,
            )
            rss = compute_rss(estimator, graph)
            mse = compute_mse(estimator, graph)
            # score = compute_log_lik(estimator, graph)
            score = np.sum(estimator.score_samples(graph, clip=c))
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


def select_rdpg(graph, n_components_try_range, directed):
    if is_symmetric(graph) and directed:
        msg = (
            "select_RDPG was given an undirected graph but you wanted"
            + " a directed model"
        )
        raise ValueError(msg)
    columns = ["rss", "mse", "score", "n_components_try", "n_params", "directed"]
    out_df = pd.DataFrame(
        np.nan, index=range(len(n_components_try_range)), columns=columns
    )
    c = 1 / (graph.size - graph.shape[0])
    for i, n_components in enumerate(n_components_try_range):
        estimator, n_params = estimate_rdpg(graph, n_components=n_components)
        rss = compute_rss(estimator, graph)
        mse = compute_mse(estimator, graph)
        # score = compute_log_lik(estimator, graph)

        score = np.sum(estimator.score_samples(graph, clip=c))
        out_df.loc[i, "n_params"] = n_params
        out_df.loc[i, "rss"] = rss
        out_df.loc[i, "mse"] = mse
        out_df.loc[i, "score"] = score
        out_df.loc[i, "n_components_try"] = n_components
        out_df.loc[i, "directed"] = directed

    return out_df

