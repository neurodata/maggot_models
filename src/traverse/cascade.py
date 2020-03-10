from anytree import Node
import numpy as np
from anytree import LevelOrderGroupIter, Node


def generate_cascade_paths(start_ind, probs, depth, stop_inds=[], max_depth=10):
    if (depth < max_depth) and (start_ind not in stop_inds):
        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        paths = []
        for i in next_inds:
            next_paths = generate_cascade_paths(
                i, probs, depth + 1, stop_inds=stop_inds, max_depth=max_depth
            )
            for p in next_paths:
                p.insert(0, start_ind)
                paths.append(p)
        return paths
    else:
        return [[start_ind]]


def generate_cascade_tree(
    node, probs, depth, stop_inds=[], visited=[], max_depth=10, loops=False
):
    start_ind = node.name
    if (depth < max_depth) and (start_ind not in stop_inds):
        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        for n in next_inds:
            if n not in visited:
                next_node = Node(n, parent=node)
                visited.append(n)
                generate_cascade_tree(
                    next_node,
                    probs,
                    depth + 1,
                    stop_inds=stop_inds,
                    visited=visited,
                    max_depth=max_depth,
                )
    return node


def cascades_from_node(
    start_ind,
    probs,
    stop_inds=[],
    max_depth=10,
    n_sims=1000,
    seed=None,
    n_bins=None,
    method="tree",
):
    if n_bins is None:
        n_bins = max_depth
    np.random.seed(seed)
    n_verts = len(probs)
    node_hist_mat = np.zeros((n_verts, n_bins), dtype=int)
    for n in range(n_sims):
        if method == "tree":
            _cascade_tree_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat)
        elif method == "path":
            _cascade_path_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat)
    return node_hist_mat


def _cascade_tree_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat):
    root = Node(start_ind)
    root = generate_cascade_tree(
        root, probs, 1, stop_inds=stop_inds, visited=[], max_depth=max_depth
    )
    for level, children in enumerate(LevelOrderGroupIter(root)):
        for node in children:
            node_hist_mat[node.name, level] += 1


def _cascade_path_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat):
    paths = generate_cascade_paths(
        start_ind, probs, 1, stop_inds=stop_inds, max_depth=max_depth
    )
    for path in paths:
        for i, node in enumerate(path):
            node_hist_mat[node, i] += 1
