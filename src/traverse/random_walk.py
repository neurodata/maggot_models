import numpy as np
from .traverse import BaseTraverse


class RandomWalk(BaseTraverse):
    def _choose_next(self):
        node_transition_probs = self.transition_probs[self._active]
        if np.isclose(np.sum(node_transition_probs), 1):
            nxt = np.random.choice(self.n_verts, p=node_transition_probs)
            return nxt


def to_markov_matrix(adj):
    prob_mat = adj.copy()
    row_sums = prob_mat.sum(axis=1)
    row_sums[row_sums == 0] = 1  # plug the holes
    prob_mat = prob_mat / row_sums[:, np.newaxis]
    return prob_mat


def random_walks_from_node(
    start_ind,
    probs,
    stop_inds=[],
    max_depth=10,
    n_sims=1000,
    seed=None,
    n_bins=None,
    method="tree",
):
    pass


def generate_random_walks(
    prob_mat, from_inds, out_inds, n_walks=100, max_walk=25, return_stuck=False
):
    n_verts = len(prob_mat)
    dead_inds = np.where(prob_mat.sum(axis=1) == 0)[0]
    stop_reasons = np.zeros(4)
    sm_paths = []
    visit_orders = {i: [] for i in range(n_verts)}
    for s in from_inds:
        for n in range(n_walks):
            curr_ind = s
            n_steps = 0
            path = [s]
            visit_orders[s].append(len(path))
            while (
                (curr_ind not in out_inds)
                and (n_steps <= max_walk)
                and (curr_ind not in dead_inds)
            ):
                next_ind = np.random.choice(n_verts, p=prob_mat[curr_ind])
                n_steps += 1
                curr_ind = next_ind
                path.append(curr_ind)
                visit_orders[curr_ind].append(len(path))
            if curr_ind in out_inds:
                stop_reasons[0] += 1
                sm_paths.append(path)
            elif curr_ind in dead_inds:
                stop_reasons[1] += 1
                if return_stuck:
                    sm_paths.append(path)
            elif n_steps > max_walk:
                stop_reasons[2] += 1
            else:
                stop_reasons[3] += 1

    print(stop_reasons / stop_reasons.sum())
    print(len(sm_paths))
    return sm_paths, visit_orders


# def _step(start, prob_mat, path, max_walk=30):
#     if len(path) < max_walk:
#         choice = np.random.choice(len(prob_mat), p=prob_mat[start])
#         path.append(choice)
#         return choice

# def _random_walk(start, prob_mat, )
#     curr = start


# if curr_ind in out_inds:
#     stop_reasons[0] += 1
#     sm_paths.append(path)
# if curr_ind in dead_inds:
#     stop_reasons[1] += 1
# if n_steps > max_walk:
#     stop_reasons[2] += 1

