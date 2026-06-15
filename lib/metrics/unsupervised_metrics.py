import numpy as np

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)


class UnsupervisedMetrics:
    """
    Classe per la valutazione di modelli di clustering.

    Supporta:
    - Silhouette Score (senza label)
    - Adjusted Rand Index (con label reali)
    - Normalized Mutual Information
    - Matrice di contingenza
    """

    def __init__(self):
        self.silhouette = None
        self.ari = None
        self.nmi = None
        self.contingency_matrix = None

    def evaluate(self, X, labels_pred, y_true=None):
        """
        Calcola tutte le metriche disponibili.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature space usato per clustering
        labels_pred : array-like
            Cluster assegnati dal modello
        y_true : array-like, optional
            Label reali (se disponibili)
        """

        # Silhouette Score (sempre disponibile)
        if len(set(labels_pred)) > 1:
            self.silhouette = silhouette_score(X, labels_pred)
        else:
            self.silhouette = -1  # clustering non valido

        # Metriche supervisionate (solo se y_true presente)
        if y_true is not None:
            self.ari = adjusted_rand_score(y_true, labels_pred)

            self.nmi = normalized_mutual_info_score(
                y_true,
                labels_pred
            )

            self.contingency_matrix = confusion_matrix(
                y_true,
                labels_pred
            )

        return self

    def report(self):
        """
        Ritorna un dizionario con tutte le metriche.
        """

        return {
            "silhouette_score": self.silhouette,
            "adjusted_rand_index": self.ari,
            "normalized_mutual_info": self.nmi,
            "contingency_matrix": self.contingency_matrix
        }

    def print_report(self):
        """
        Stampa formattata delle metriche.
        """

        print("\n=== UNSUPERVISED EVALUATION ===")

        print(f"Silhouette Score: {self.silhouette}")

        if self.ari is not None:
            print(f"ARI: {self.ari}")

        if self.nmi is not None:
            print(f"NMI: {self.nmi}")

        if self.contingency_matrix is not None:
            print("\nContingency Matrix:")
            print(self.contingency_matrix)