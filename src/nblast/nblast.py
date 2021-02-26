import numpy as np
from sklearn.preprocessing import QuantileTransformer
from graspologic.utils import pass_to_ranks, symmetrize
from scipy.stats.mstats import gmean


def preprocess_nblast(
    nblast_scores, symmetrize_mode="geom", transform="ptr", return_untransformed=False
):
    distance = nblast_scores  # the raw nblast scores are dissimilarities/distances
    indices = np.triu_indices_from(distance, k=1)

    if symmetrize_mode == "geom":
        fwd_dist = distance[indices]
        back_dist = distance[indices[::-1]]
        stack_dist = np.concatenate(
            (fwd_dist.reshape(-1, 1), back_dist.reshape(-1, 1)), axis=1
        )
        geom_mean = gmean(stack_dist, axis=1)
        sym_distance = np.zeros_like(distance)
        sym_distance[indices] = geom_mean
        sym_distance[indices[::-1]] = geom_mean
    else:  # simple average
        sym_distance = symmetrize(distance)

    # make the distances between 0 and 1
    sym_distance /= sym_distance.max()
    sym_distance -= sym_distance.min()
    # and then convert to similarity
    morph_sim = 1 - sym_distance

    if transform == "quantile":
        quant = QuantileTransformer(n_quantiles=2000)
        transformed_vals = quant.fit_transform(morph_sim[indices].reshape(-1, 1))
        transformed_vals = np.squeeze(transformed_vals)
        transformed_morph = np.ones_like(morph_sim)
        transformed_morph[indices] = transformed_vals
        transformed_morph[indices[::-1]] = transformed_vals
    elif transform == "ptr":
        transformed_morph = pass_to_ranks(morph_sim)
        np.fill_diagonal(
            transformed_morph, 1
        )  # should be exactly 1, isnt cause of ties
    elif transform == "log":
        raise NotImplementedError()
    else:
        transformed_morph = morph_sim
    if return_untransformed:
        return transformed_morph, morph_sim
    else:
        return transformed_morph