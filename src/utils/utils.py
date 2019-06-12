import numpy as np
from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import RDPGEstimator, SBEstimator
from graspy.simulations import p_from_latent, sample_edges


def estimate_assignments(graph, n_communities, n_components=None):
    embed_graph = graph.copy()
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(
        embed_graph
    )
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


def estimate_sbm(graph, n_communities, n_components=None):
    vertex_assignments, n_params = estimate_assignments(
        graph, n_communities, n_components
    )
    estimator = SBEstimator(directed=False, loops=False)
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
    p_mat = estimator.p_mat_
    diff = (p_mat - graph) ** 2
    rss = np.sum(diff)
    return rss


def compute_log_lik(estimator, graph, c=0):
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
