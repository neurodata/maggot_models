import numpy as np
from sklearn.base import BaseEstimator
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD


def unscale(X):
    # TODO implement as a setting in graspologic
    norms = np.linalg.norm(X, axis=0)
    X = X / norms[None, :]
    return X


def _check_matrices(matrices):
    if isinstance(matrices, np.ndarray) and (matrices.ndim == 2):
        matrices = [matrices]
    return matrices


class JointEmbed(BaseEstimator):
    def __init__(
        self,
        n_components=None,
        stage1_n_components=None,
        method="ase",
        scaled=False,
        algorithm="randomized",
        embed_kws={},
    ):
        self.n_components = n_components
        self.stage1_n_components = stage1_n_components
        self.embed_kws = embed_kws
        self.method = method
        self.scaled = scaled
        self.algorithm = algorithm
        self.embed_kws = embed_kws

    def _embed_matrices(self, matrices):
        embeddings = []
        models = []
        for matrix in matrices:
            model = self._Embed(n_components=self.stage1_n_components, **self.embed_kws)
            embedding = model.fit_transform(matrix)
            embeddings.append(embedding)
            models.append(model)
        return embeddings, models

    def fit_transform(self, graphs, similarities, weights=None):
        if self.method == "ase":
            self._Embed = AdjacencySpectralEmbed
        elif self.method == "lse":
            self._Embed = LaplacianSpectralEmbed
            if self.embed_kws == {}:
                print("here")
                self.embed_kws["form"] = "R-DAD"

        graphs = _check_matrices(graphs)
        similarities = _check_matrices(similarities)

        if weights is None:
            weights = np.ones(len(graphs) + len(similarities))

        # could all be in one call, but want to leave open the option of treating the
        # graphs vs the similarities separately.
        graph_embeddings, graph_models = self._embed_matrices(graphs)
        similarity_embeddings, similarity_models = self._embed_matrices(similarities)

        self.graph_models_ = graph_models
        self.similarity_models_ = similarity_models

        embeddings = graph_embeddings + similarity_embeddings
        concat_embeddings = []
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, tuple):
                embedding = (embedding,)
            for e in embedding:
                if not self.scaled:
                    e = unscale(e)
                concat_embeddings.append(weights[i] * e)

        concat_embeddings = np.concatenate(concat_embeddings, axis=1)

        joint_embedding, joint_singular_values, _ = selectSVD(
            concat_embeddings, n_components=self.n_components, algorithm=self.algorithm
        )

        self.singular_values_ = joint_singular_values
        return joint_embedding
