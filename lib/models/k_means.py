import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from lib.models.base import AbstractModel


_DEFAULT_PARAM_GRID = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8],
    "init": ["k-means++"],
    "n_init": [10],
    "max_iter": [300]
}


class GenericKMeans(AbstractModel):

    def train(
        self,
        X_train,
        random_state=42,
        param_grid=None
    ):
        """
        Addestra K-Means cercando la combinazione di parametri
        che massimizza il Silhouette Score.
        """

        param_grid = param_grid or _DEFAULT_PARAM_GRID

        best_score = -1
        best_model = None
        best_params = None

        for n_clusters in param_grid["n_clusters"]:
            for init in param_grid["init"]:
                for n_init in param_grid["n_init"]:
                    for max_iter in param_grid["max_iter"]:

                        model = KMeans(
                            n_clusters=n_clusters,
                            init=init,
                            n_init=n_init,
                            max_iter=max_iter,
                            random_state=random_state
                        )

                        with joblib.parallel_backend("threading"):
                            labels = model.fit_predict(X_train)

                        # silhouette richiede almeno 2 cluster
                        if len(set(labels)) > 1:
                            score = silhouette_score(
                                X_train,
                                labels
                            )

                            if score > best_score:
                                best_score = score
                                best_model = model
                                best_params = {
                                    "n_clusters": n_clusters,
                                    "init": init,
                                    "n_init": n_init,
                                    "max_iter": max_iter
                                }

        self.model = best_model
        self.best_params = best_params
        self.best_score = best_score

