"""
Created on Thu Mar 14 14:19:55 2019
These functions allow for "brute clustering," inspired by R's mclust.
Clustering is performed first by hierarchical agglomeration, then fitting a
Gaussian Mixture via Expectation Maximization (EM). There are several ways to 
perform both agglomeration and EM so these functions performs the (specified)
combinations of methods then evaluates each according to BIC.
@author: Thomas Athey
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal

"""
Calculates likelihood of a set of data from a GMM, then calculates BIC
Inputs:
    x - nxd datapoints
    wts - list of mixture weights (same length as means and variances)
    means - list of d numpy arrays of mixture means
    variances - list of dxd covariance matrices
    k - number of parameters
    
Outputs:
    bic - BIC where higher is better
"""


def calcBIC(x, wts, means, variances, k):
    n = x.shape[0]
    likelihood = 0
    for wt, mu, var in zip(wts, means, variances):
        mu = np.squeeze(mu)
        var = np.squeeze(var)
        try:
            var = multivariate_normal(mu, var)
        except np.linalg.LinAlgError:
            return -np.inf
        likelihood += wt * var.pdf(x)
    loglik = np.sum(np.log(likelihood))
    bic = 2 * loglik - np.log(n) * k
    return bic


"""
Calculates BIC from input that is formatted either as the sklearn GaussianMixture
components or from data that was saved to a csv in R
 
Inputs
    data - nxd numpy array of data
    wts - k numpy array of mixture weights
    mus - kxd numpy array of means
    covs - kxdxd in the case of r and in python, the shape depends on the model type
        (see GaussianMixture class)
    m - a string that specifies the model, and implies that format of the other inputs
        (e.g. 'VII' implies that the parameters were read from a csv that was written by R)
Outputs
    BIC - bic value as calculated by the function above
"""


def processBIC(data, wts, mus, covs, m):
    d = data.shape[1]
    k = len(wts)

    # These options indicate mclust model types, so the format of covs is how
    # it was written to a csv in R
    if m == "VII":
        params = k * (1 + d + 1)
        covs = np.split(covs, covs.shape[0])
    elif m == "EEE":
        params = k * (1 + d) + d * (d + 1) / 2
        covs = np.split(covs, covs.shape[0])
    elif m == "VVV":
        params = k * (1 + d + d * (d + 1) / 2)
        covs = np.split(covs, covs.shape[0])
    elif m == "VVI":
        params = k * (1 + d + d)
        covs = np.split(covs, covs.shape[0])
    # These options indicate GaussianMixture types, so the format of covs is
    # sklearrn.mixture.GaussianMixture.covariances_
    elif m == "spherical":
        params = k * (1 + d + 1)
        covs = [v * np.identity(d) for v in covs]
    elif m == "tied":
        params = k * (1 + d) + d * (d + 1) / 2
        covs = [covs for v in np.arange(k)]
    elif m == "full":
        params = k * (1 + d + d * (d + 1) / 2)
        covs = np.split(covs, covs.shape[0])
    elif m == "diag":
        params = k * (1 + d + d)
        covs = [np.diag(covs[i, :]) for i in np.arange(k)]

    params = params - 1  # because the weights must add to 1
    wts = np.split(wts, wts.shape[0])
    means = np.split(mus, mus.shape[0])
    return calcBIC(data, wts, means, covs, params)


colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "yellow",
    "black",
    "brown",
    "lightsalmon",
    "greenyellow",
    "cornflowerblue",
    "tan",
    "violet",
    "gold",
    "slategray",
    "peru",
    "indianred",
    "darkolivegreen",
    "navy",
    "darkgoldenrod",
    "deeppink",
    "darkkhaki",
    "silver",
    "saddlebrown",
]


def agglomerate(data, aff, link, k):
    """
    Hierarchical Agglomeration
    inputs:
        data - nxd numpy array
        aff - affinity technique, an element of ['euclidean','manhattan','cosine']
        link - linkage technique, an element of ['ward','complete','average','single']
        k - number of clusters
    outputs:
        one_hot - nxk numpy array with a single one in each row indicating cluster
            membership
    exceptions:
        ward linkage can only be used with euclidean/l2 affinity so if ward is 
        specified with a different linkage then there is an Exception
    """
    n = data.shape[0]
    if link == "ward" and aff != "euclidean":
        raise Exception("Ward linkage is only valid with Euclidean affinity")
    agglom = AgglomerativeClustering(n_clusters=k, affinity=aff, linkage=link).fit(data)
    one_hot = np.zeros([n, k])
    one_hot[np.arange(n), agglom.labels_] = 1
    return one_hot


def initialize_params(data, one_hot, cov):
    """
    sklearn's Gaussian Mixture does not allow initialization from class membership
    but it does allow from initialization of mixture parameters, so here we calculate
    the mixture parameters according to class membership
    input:
        data - nxd numpy array 
        one_hot - nxd numpy array with a single one in each row indicating cluster
            membership
        k - number of clusters
    output:
        weights - k array of mixing weights
        means - kxd array of means of mixture components
        precisions - precision matrices, format depends on the EM clustering option
            (eg 'full' mode needs a list of matrices, one for each mixture
            component,but 'tied' mode only needs a single matrix, since all
            precisions are constrained to be equal)
    """

    n = data.shape[0]
    weights, means, covariances = _estimate_gaussian_parameters(
        data, one_hot, 1e-06, cov
    )
    weights /= n

    precisions_cholesky_ = _compute_precision_cholesky(covariances, cov)
    if cov == "tied":
        c = precisions_cholesky_
        precisions = np.dot(c, c.T)
    elif cov == "diag":
        precisions = precisions_cholesky_
    else:
        precisions = [np.dot(c, c.T) for c in precisions_cholesky_]

    return weights, means, precisions


def cluster(data, aff, link, cov, k, c_true=None):
    """
    Cluster according to specified method
    input:
        data - nxk numpy matrix of data
        c_true - n array of true cluster membership
        aff - affinity, element of ['euclidean','manhattan','cosine'] or none for EM from scratch
        link - linkage, element of ['ward','complete','average','single'], or none for EM from scratch
        cov - covariance, element of ['full','tied','diag','spherical']
        k - # of clusters
    output:
        c_hat - n array of clustering results
        means - kxd array of means of mixture components
        bic - Bayes Information Criterion for this clustering
        ari - Adjusted Rand Index to comparing clustering result to true clustering
        reg - regularization parameter that was used in the clustering results
            (0 or 1e-6)
    """
    iter_num = 100
    if aff == "none" or link == "none":
        try:  # no regularization
            reg = 0
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                reg_covar=reg,
                max_iter=iter_num,
                verbose=0,
                verbose_interval=1,
            )
            c_hat = gmm.fit_predict(data)
            bic = processBIC(data, gmm.weights_, gmm.means_, gmm.covariances_, cov)
            if any([sum(c_hat == i) <= 1 for i in range(k)]) or bic == -np.inf:
                raise ValueError
        # if there was a numerical error during EM,or while calculating BIC,
        # or if the clustering found a class with only one element
        except:  # regularize
            reg = 1e-6
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                reg_covar=reg,
                max_iter=iter_num,
                verbose=0,
                verbose_interval=1,
            )
            c_hat = gmm.fit_predict(data)
            bic = processBIC(data, gmm.weights_, gmm.means_, gmm.covariances_, cov)
    else:
        one_hot = agglomerate(data, aff, link, k)
        weights, means, precisions = initialize_params(data, one_hot, cov)

        try:
            reg = 0
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                weights_init=weights,
                means_init=means,
                precisions_init=precisions,
                max_iter=iter_num,
                reg_covar=reg,
                verbose=0,
                verbose_interval=1,
            )
            c_hat = gmm.fit_predict(data)
            bic = processBIC(data, gmm.weights_, gmm.means_, gmm.covariances_, cov)
            if any([sum(c_hat == i) <= 1 for i in range(k)]) or bic == -np.inf:
                raise ValueError
        # if there was a numerical error, or if initial clustering produced a
        # mixture component with only one element
        except:
            reg = 1e-6
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                weights_init=weights,
                means_init=means,
                precisions_init=precisions,
                max_iter=iter_num,
                reg_covar=reg,
                verbose=0,
                verbose_interval=1,
            )
            c_hat = gmm.fit_predict(data)
            bic = processBIC(data, gmm.weights_, gmm.means_, gmm.covariances_, cov)

    if c_true is not None:
        ari = adjusted_rand_score(c_true, c_hat)
    else:
        ari = None

    means = gmm.means_
    return c_hat, means, bic, ari, reg, gmm._n_parameters()


def brute_cluster(
    x,
    ks,
    affinities=None,
    linkages=None,
    covariance_types=None,
    c_true=None,
    plot=True,
    savefigs=True,
    verbose=0,
):
    """
    Cluster all combinations of options and plot results
    inputs:
        x - nxd array of data
        c_true - n array of true clustering
        affinites - list of affinity modes, each must be an element of
            ['none,'euclidean','manhattan','cosine']
        linkages - list of linkage modes, each must be an element of
            ['none','ward','complete','average','single']
        covariance_types - list of covariance modes, each must be an element of
            ['full','tied','diag','spherical']
        ks - list of cluster numbers
        savefigs - None indicates that figures should not be saved, a string value
            indicates the name that should be used when saving the figures
        verbose - if 0, no output, if 1, output the current clustering options
            being used
    outputs:
        bics,aris - 44xlength(ks) array of bic and ari values for each clustering result
    """
    if affinities is None:
        affinities = ["none", "euclidean", "manhattan", "cosine"]
    if linkages is None:
        linkages = ["none", "ward", "complete", "average", "single"]
    if covariance_types is None:
        covariance_types = ["full", "tied", "diag", "spherical"]

    cov_dict = {"full": 0, "tied": 1, "diag": 2, "spherical": 3}
    aff_dict = {"none": 0, "euclidean": 0, "manhattan": 1, "cosine": 2}
    link_dict = {"none": 0, "ward": 1, "complete": 2, "average": 3, "single": 4}

    # 11 agglomeration combos: 4 with l2 affinity, 3 with l1, 3 with cos, and no agglom
    # 4 EM options: full, tied, diag, spherical
    bics = np.zeros([44, len(ks)]) - np.inf
    aris = np.zeros([44, len(ks)]) - np.inf

    best_ari = float("-inf")
    best_bic = float("-inf")

    best_n_params = np.inf

    for i, k in enumerate(ks):
        for af in affinities:
            for li in linkages:
                # some combinations don't work, skip these
                if li == "ward" and af != "euclidean":
                    continue
                if (li == "none" and af != "none") or (af == "none" and li != "none"):
                    continue
                for cov in covariance_types:
                    if verbose == 1:
                        print(
                            "K="
                            + k
                            + " Affinity= "
                            + af
                            + " Linkage= "
                            + li
                            + " Covariance= "
                            + cov
                        )
                    row = 11 * cov_dict[cov] + 3 * aff_dict[af] + link_dict[li]

                    c_hat, means, bic, ari, reg, n_params = cluster(
                        x, af, li, cov, k, c_true
                    )

                    bics[row, i] = bic
                    aris[row, i] = ari

                    if c_true is not None and ari > best_ari:
                        best_ari = ari
                        best_combo_ari = [af, li, cov]
                        best_c_hat_ari = c_hat
                        best_k_ari = k

                    if bic > best_bic:
                        best_bic = bic
                        best_combo_bic = [af, li, cov]
                        best_c_hat_bic = c_hat
                        best_k_bic = k
                        best_ari_bic = ari
                        best_means_bic = means
                        reg_bic = reg
                        best_n_params = n_params

    # True plot**********************************
    if plot and c_true is not None:
        plt.figure(figsize=(8, 8))
        ptcolors = [colors[i] for i in c_true.astype(int)]
        plt.scatter(x[:, 0], x[:, 1], c=ptcolors)
        plt.title("True labels")
        plt.xlabel("First feature")
        plt.ylabel("Second feature")
        if savefigs is not None:
            plt.savefig(savefigs + "_python_true.jpg")

    # Plot with best BIC*********************************
    if plot:
        plt.figure(figsize=(8, 8))
        # ptcolors = [colors[i] for i in best_c_hat_bic]
        plt.scatter(x[:, 0], x[:, 1], c=best_c_hat_bic)
        # mncolors = [colors[i] for i in np.arange(best_k_bic)]
        mncolors = [i for i in np.arange(best_k_bic)]
        plt.scatter(best_means_bic[:, 0], best_means_bic[:, 1], c=mncolors, marker="x")
        plt.title(
            "py(agg-gmm) BIC %3.0f from " % best_bic
            + str(best_combo_bic)
            + " k="
            + str(best_k_bic)
            + " reg="
            + str(reg_bic)
        )  # + "iter=" + str(best_iter_bic))
        plt.legend()
        plt.xlabel("First feature")
        plt.ylabel("Second feature")
        if savefigs is not None:
            plt.savefig(savefigs + "_python_bestbic.jpg")

    titles = ["full", "tied", "diag", "spherical"]

    if plot and c_true is not None:
        # Plot with best ARI************************************
        plt.figure(figsize=(8, 8))
        ptcolors = [colors[i] for i in best_c_hat_ari]
        plt.scatter(x[:, 0], x[:, 1], c=ptcolors)
        plt.title(
            "py(agg-gmm) ARI %3.3f from " % best_ari
            + str(best_combo_ari)
            + " k="
            + str(best_k_ari)
        )  # + "iter=" + str(best_iter_ari))
        plt.xlabel("First feature")
        plt.ylabel("Second feature")
        if savefigs is not None:
            plt.savefig(savefigs + "_python_bestari.jpg")

        # ARI vs BIC********************************
        plt.figure(figsize=(8, 8))
        for row in np.arange(4):
            xs = bics[row * 11 : (row + 1) * 11, :]
            ys = aris[row * 11 : (row + 1) * 11, :]
            idxs = (xs != -np.inf) * (ys != -np.inf)
            plt.scatter(xs[idxs], ys[idxs], label=titles[row])

        idxs = (bics != -np.inf) * (aris != -np.inf)
        slope, _, r_value, _, p_value = stats.linregress(bics[idxs], aris[idxs])
        plt.xlabel("BIC")
        plt.ylabel("ARI")
        plt.legend(loc="lower right")
        plt.title(
            "Pyclust's ARI vs BIC for Drosophila Data with Correlation r^2=%2.2f"
            % (r_value ** 2)
        )
        if savefigs is not None:
            plt.savefig(savefigs + "_python_bicari.jpg")

    if plot:
        # plot of all BICS*******************************
        labels = {
            0: "none",
            1: "l2/ward",
            2: "l2/complete",
            3: "l2/average",
            4: "l2/single",
            5: "l1/complete",
            6: "l1/average",
            7: "l1/single",
            8: "cos/complete",
            9: "cos/average",
            10: "cos/single",
        }

        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            2, 2, sharey="row", sharex="col", figsize=(10, 10)
        )
        for row in np.arange(bics.shape[0]):
            if all(bics[row, :] == -np.inf):
                continue
            if row <= 10:
                ax0.plot(np.arange(1, len(ks) + 1), bics[row, :])
            elif row <= 21:
                ax1.plot(
                    np.arange(1, len(ks) + 1), bics[row, :], label=labels[row % 11]
                )
            elif row <= 32:
                ax2.plot(np.arange(1, len(ks) + 1), bics[row, :])
            elif row <= 43:
                ax3.plot(np.arange(1, len(ks) + 1), bics[row, :])

        ax0.set_title(titles[0])
        ax0.set(ylabel="bic")
        ax1.set_title(titles[1])
        ax1.legend(loc="lower right")
        ax2.set_title(titles[2])
        ax2.set(xlabel="k")
        ax2.set(ylabel="bic")
        ax3.set_title(titles[3])
        ax3.set(xlabel="k")
        if savefigs is not None:
            plt.savefig(savefigs + "_python_bicplot.jpg")

    return best_c_hat_bic, best_n_params

