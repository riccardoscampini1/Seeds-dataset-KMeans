import pandas as pd
import numpy as np


class FeatureEngineering:
    """
    Classe per creare nuove feature a partire da feature esistenti.
    """

    def __init__(self):
        pass

    def create_ratio_feature(
        self,
        X: pd.DataFrame,
        numerator: str,
        denominator: str,
        new_feature_name: str = None,
        drop_originals: bool = True,
    ) -> pd.DataFrame:
        """
        Crea una feature come rapporto tra due colonne.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset di input.
        numerator : str
            Colonna al numeratore.
        denominator : str
            Colonna al denominatore.
        new_feature_name : str, optional
            Nome della nuova feature.
        drop_originals : bool, default=True
            Se True elimina le colonne originali.

        Returns
        -------
        pd.DataFrame
            Nuovo dataset con la feature creata.
        """

        if numerator not in X.columns:
            raise ValueError(
                f"Column '{numerator}' not found."
            )

        if denominator not in X.columns:
            raise ValueError(
                f"Column '{denominator}' not found."
            )

        X_new = X.copy()

        if new_feature_name is None:
            new_feature_name = (
                f"{numerator}_over_{denominator}"
            )

        X_new[new_feature_name] = (
            X_new[numerator] /
            X_new[denominator].replace(0, np.nan)
        )

        X_new[new_feature_name] = (
            X_new[new_feature_name]
            .replace([np.inf, -np.inf], np.nan)
        )

        if drop_originals:
            X_new = X_new.drop(
                columns=[numerator, denominator]
            )

        return X_new