import numpy as np


def generate_random_cascade(start_ind, probs, depth, stop_inds=[], max_depth=10):
    if (depth < max_depth) and (start_ind not in stop_inds):
        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        paths = []
        for i in next_inds:
            next_paths = generate_random_cascade(
                i, probs, depth + 1, stop_inds=stop_inds, max_depth=max_depth
            )
            for p in next_paths:
                p.insert(0, start_ind)
                paths.append(p)
        return paths
    else:
        return [[start_ind]]
