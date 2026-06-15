"""
Multicollinearity preprocessing utilities.
"""

from collections import defaultdict, deque
from typing import Optional

import numpy as np
import pandas as pd

from lib.multicollinearity import MulticollinearityHandler
from src.utils import PreprocessingError, get_logger

logger = get_logger(__name__)


class MulticollinearityReducer:
    """
    Remove highly correlated variables while keeping the least redundant feature.

    The underlying correlated-pair detection is delegated to
    lib.multicollinearity.MulticollinearityHandler. For each connected group of
    variables with absolute correlation above the threshold, this reducer keeps
    the feature with the lowest mean absolute correlation to the other numeric
    variables and drops the rest.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.handler = MulticollinearityHandler()
        self.correlated_pairs = []
        self.cols_to_drop = []
        self.cols_to_keep = []
        self.mean_abs_correlations = {}
        self.report_ = {}

    def fit(self, df: pd.DataFrame) -> "MulticollinearityReducer":
        """Fit the reducer and decide which columns should be removed."""
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError("Expected a pandas DataFrame.")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            self._set_empty_report()
            return self

        self.handler.configure(threshold=self.threshold, drop=False)
        self.handler.fit(numeric_df)
        lib_report = self.handler.report()
        self.correlated_pairs = lib_report["pairs"]

        corr = numeric_df.corr().abs()
        self.mean_abs_correlations = {
            col: float(corr[col].drop(index=col).mean())
            for col in corr.columns
        }

        groups = self._build_correlated_groups()
        cols_to_drop = set()
        cols_to_keep = set()

        for group in groups:
            keep_col = min(
                group,
                key=lambda col: (self.mean_abs_correlations[col], col),
            )
            cols_to_keep.add(keep_col)
            cols_to_drop.update(col for col in group if col != keep_col)

        self.cols_to_drop = sorted(cols_to_drop)
        self.cols_to_keep = sorted(cols_to_keep)
        self.report_ = {
            "threshold": self.threshold,
            "pairs": self.correlated_pairs,
            "cols_to_keep": self.cols_to_keep,
            "cols_to_drop": self.cols_to_drop,
            "mean_abs_correlations": {
                col: round(value, 4)
                for col, value in self.mean_abs_correlations.items()
            },
        }

        logger.info(
            f"Multicollinearity reduction fitted: "
            f"{len(self.correlated_pairs)} correlated pairs, "
            f"{len(self.cols_to_drop)} columns to drop"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the selected multicollinear columns."""
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError("Expected a pandas DataFrame.")

        return df.drop(
            columns=[col for col in self.cols_to_drop if col in df.columns],
            errors="ignore",
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the reducer and transform the dataframe."""
        return self.fit(df).transform(df)

    def report(self) -> dict:
        """Return the fitted multicollinearity report."""
        return self.report_.copy()

    def _build_correlated_groups(self) -> list[set[str]]:
        graph = defaultdict(set)
        for pair in self.correlated_pairs:
            col_a = pair["col_a"]
            col_b = pair["col_b"]
            graph[col_a].add(col_b)
            graph[col_b].add(col_a)

        groups = []
        visited = set()

        for start in graph:
            if start in visited:
                continue

            group = set()
            queue = deque([start])
            visited.add(start)

            while queue:
                col = queue.popleft()
                group.add(col)
                for neighbor in graph[col]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            groups.append(group)

        return groups

    def _set_empty_report(self) -> None:
        self.correlated_pairs = []
        self.cols_to_drop = []
        self.cols_to_keep = []
        self.mean_abs_correlations = {}
        self.report_ = {
            "threshold": self.threshold,
            "pairs": [],
            "cols_to_keep": [],
            "cols_to_drop": [],
            "mean_abs_correlations": {},
        }
