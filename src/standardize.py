import numpy as np

class FeatureStandardizer:
    """
    Simple per-dimension standardizer (z-score) with small epsilon.
    Fit on training set, then reuse on val/test to ensure consistency.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> None:
        assert X.ndim == 2, "Expected X to be 2D (N, D)"
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + self.eps

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "Standardizer not fitted"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)