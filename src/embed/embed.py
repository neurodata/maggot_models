import numpy as np
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from graspy.utils import pass_to_ranks, get_lcc


def ase(adj, n_components, ptr=True):
    if ptr:
        adj = pass_to_ranks(adj)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    latent = ase.fit_transform(adj)
    latent = np.concatenate(latent, axis=-1)
    return latent


def to_laplace(graph, form="DAD", regularizer=None):
    r"""
    A function to convert graph adjacency matrix to graph laplacian. 
    Currently supports I-DAD, DAD, and R-DAD laplacians, where D is the diagonal
    matrix of degrees of each node raised to the -1/2 power, I is the 
    identity matrix, and A is the adjacency matrix.
    
    R-DAD is regularized laplacian: where :math:`D_t = D + regularizer*I`.
    Parameters
    ----------
    graph: object
        Either array-like, (n_vertices, n_vertices) numpy array,
        or an object of type networkx.Graph.
    form: {'I-DAD' (default), 'DAD', 'R-DAD'}, string, optional
        
        - 'I-DAD'
            Computes :math:`L = I - D*A*D`
        - 'DAD'
            Computes :math:`L = D*A*D`
        - 'R-DAD'
            Computes :math:`L = D_t*A*D_t` where :math:`D_t = D + regularizer*I`
    regularizer: int, float or None, optional (default=None)
        Constant to be added to the diagonal of degree matrix. If None, average 
        node degree is added. If int or float, must be >= 0. Only used when 
        ``form`` == 'R-DAD'.
    Returns
    -------
    L: numpy.ndarray
        2D (n_vertices, n_vertices) array representing graph 
        laplacian of specified form
    References
    ----------
    .. [1] Qin, Tai, and Karl Rohe. "Regularized spectral clustering
           under the degree-corrected stochastic blockmodel." In Advances
           in Neural Information Processing Systems, pp. 3120-3128. 2013
    """
    valid_inputs = ["I-DAD", "DAD", "R-DAD"]
    if form not in valid_inputs:
        raise TypeError("Unsuported Laplacian normalization")

    A = graph

    in_degree = np.sum(A, axis=0)
    out_degree = np.sum(A, axis=1)

    # regularize laplacian with parameter
    # set to average degree
    if form == "R-DAD":
        if regularizer is None:
            regularizer = 1
        elif not isinstance(regularizer, (int, float)):
            raise TypeError(
                "Regularizer must be a int or float, not {}".format(type(regularizer))
            )
        elif regularizer < 0:
            raise ValueError("Regularizer must be greater than or equal to 0")
        regularizer = regularizer * np.mean(out_degree)

        in_degree += regularizer
        out_degree += regularizer

    with np.errstate(divide="ignore"):
        in_root = 1 / np.sqrt(in_degree)  # this is 10x faster than ** -0.5
        out_root = 1 / np.sqrt(out_degree)

    in_root[np.isinf(in_root)] = 0
    out_root[np.isinf(out_root)] = 0

    in_root = np.diag(in_root)  # just change to sparse diag for sparse support
    out_root = np.diag(out_root)

    if form == "I-DAD":
        L = np.diag(in_degree) - A
        L = in_root @ L @ in_root
    elif form == "DAD" or form == "R-DAD":
        L = out_root @ A @ in_root
    # return symmetrize(L, method="avg")  # sometimes machine prec. makes this necessary
    return L


def lse(adj, n_components, regularizer=None, ptr=True):
    if ptr:
        adj = pass_to_ranks(adj)
    lap = to_laplace(adj, form="R-DAD", regularizer=regularizer)
    ase = AdjacencySpectralEmbed(n_components=n_components, diag_aug=False)
    latent = ase.fit_transform(lap)
    # latent = LaplacianSpectralEmbed(
    #     form="R-DAD", n_components=n_components, regularizer=regularizer
    # )
    latent = np.concatenate(latent, axis=-1)
    return latent


def omni(adjs, n_components, ptr=True):
    if ptr:
        adjs = [pass_to_ranks(a) for a in adjs]
    omni = OmnibusEmbed(n_components=n_components // len(adjs))
    latent = omni.fit_transform(adjs)
    latent = np.concatenate(latent, axis=-1)  # first is for in/out
    latent = np.concatenate(latent, axis=-1)  # second is for concat. each graph
    return latent


def ase_concatenate(adjs, n_components, ptr=True):
    if ptr:
        adjs = [pass_to_ranks(a) for a in adjs]
    ase = AdjacencySpectralEmbed(n_components=n_components // len(adjs))
    graph_latents = []
    for a in adjs:
        latent = ase.fit_transform(a)
        latent = np.concatenate(latent, axis=-1)
        graph_latents.append(latent)
    latent = np.concatenate(graph_latents, axis=-1)
    return latent


def preprocess_graph(adj, *args):
    degrees = adj.sum(axis=0) + adj.sum(axis=1)

    # remove disconnected nodes
    adj, lcc_inds = get_lcc(adj, return_inds=True)

    new_args = []
    for a in args:
        new_a = a[lcc_inds]
        new_args.append(new_a)

    # remove pendants
    degrees = np.count_nonzero(adj, axis=0) + np.count_nonzero(adj, axis=1)
    not_pendant_mask = degrees != 1
    not_pendant_inds = np.array(range(len(degrees)))[not_pendant_mask]

    adj = adj[np.ix_(not_pendant_inds, not_pendant_inds)]
    new_args = []
    for a in args:
        new_a = a[not_pendant_inds]
        new_args.append(new_a)

    returns = tuple([adj] + new_args)
    return returns
