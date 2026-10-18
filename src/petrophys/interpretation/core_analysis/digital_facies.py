"""
Multi-parameter core facies classification.

Clusters depth samples using multiple core measurements
(mechanical, geochemical, textural) into discrete facies classes,
following the EpsLog/Stratum Reservoir workflow that combines grain
size, elemental concentrations, strength, and image color indices.
"""

from typing import Optional, Union, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


@dataclass
class FaciesResult:
    """
    Result of a digital core facies classification.

    Parameters
    ----------
    depth : np.ndarray
        Depth of each sample.
    facies : np.ndarray of int
        Facies class label per sample (1-based).
    n_facies : int
        Number of distinct facies.
    feature_names : list of str
        Names of the input features used for clustering.
    weights : np.ndarray
        Feature weights used during clustering.
    cluster_centers : np.ndarray
        Shape (n_facies, n_features) — cluster centroids in
        original (un-scaled) feature space.
    inertia : float
        K-means inertia (within-cluster sum of squares).
    """

    depth: np.ndarray
    facies: np.ndarray
    n_facies: int
    feature_names: List[str]
    weights: np.ndarray
    cluster_centers: np.ndarray
    inertia: float = 0.0

    @property
    def n_samples(self) -> int:
        return len(self.depth)

    def proportions(self) -> Dict[int, float]:
        """Return the fractional occurrence (0–1) of each facies."""
        total = self.n_samples
        return {
            fc: float(np.sum(self.facies == fc)) / total
            for fc in range(1, self.n_facies + 1)
        }

    def statistics(self, data: Dict[str, np.ndarray]) -> "pd.DataFrame":
        """
        Compute P13, P50, P87 of each feature per facies class.

        Parameters
        ----------
        data : dict
            Feature name → depth array (same length as self.depth).

        Returns
        -------
        pd.DataFrame
            Multi-index columns (feature, percentile) × facies rows.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        rows = []
        for fc in range(1, self.n_facies + 1):
            mask = self.facies == fc
            row: Dict[str, Any] = {
                "facies": fc,
                "count": int(np.sum(mask)),
                "proportion_pct": float(np.sum(mask)) / self.n_samples * 100,
            }
            for feat, arr in data.items():
                subset = np.asarray(arr, dtype=float)[mask]
                subset = subset[np.isfinite(subset)]
                if subset.size == 0:
                    row[f"{feat}_P13"] = row[f"{feat}_P50"] = row[f"{feat}_P87"] = float("nan")
                    continue
                row[f"{feat}_P13"] = float(np.percentile(subset, 13))
                row[f"{feat}_P50"] = float(np.percentile(subset, 50))
                row[f"{feat}_P87"] = float(np.percentile(subset, 87))
            rows.append(row)

        return pd.DataFrame(rows).set_index("facies")

    def to_dataframe(self) -> "pd.DataFrame":
        """Export depth + facies label to a DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        return pd.DataFrame({"DEPTH": self.depth, "FACIES": self.facies})

    def filter_by_depth(self, top: float, bottom: float) -> "FaciesResult":
        mask = (self.depth >= top) & (self.depth <= bottom)
        return FaciesResult(
            depth=self.depth[mask],
            facies=self.facies[mask],
            n_facies=self.n_facies,
            feature_names=self.feature_names,
            weights=self.weights.copy(),
            cluster_centers=self.cluster_centers.copy(),
            inertia=self.inertia,
        )


class DigitalCoreFacies:
    """
    K-means facies classifier for multi-parameter core data.

    Combines mechanical, geochemical, and textural measurements into
    discrete facies classes using weighted feature scaling, following
    the workflow used in EpsLog/Stratum Reservoir digital core analysis.

    Parameters
    ----------
    n_facies : int
        Number of facies classes (K in K-means).
    n_init : int
        Number of K-means initialisations (best inertia is kept).
    max_iter : int
        Maximum K-means iterations per run.
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    >>> clf = DigitalCoreFacies(n_facies=12)
    >>> result = clf.fit(
    ...     depth=depth,
    ...     data={
    ...         "D50_um":   grain_size_d50,
    ...         "UCS_MPa":  ucs,
    ...         "Al_pct":   xrf.get("Al"),
    ...         "Ca_pct":   xrf.get("Ca"),
    ...     },
    ...     weights={"D50_um": 4, "Al_pct": 2},   # rest default to 1
    ... )
    >>> stats = result.statistics(data)
    """

    def __init__(
        self,
        n_facies: int = 12,
        n_init: int = 20,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for DigitalCoreFacies: pip install scikit-learn"
            )
        self.n_facies = n_facies
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self._kmeans: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        depth: np.ndarray,
        data: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> FaciesResult:
        """
        Fit K-means on depth-aligned core measurements and return facies labels.

        Samples where any feature is NaN are excluded from fitting and
        assigned facies label 0 (unknown) in the output.

        Parameters
        ----------
        depth : np.ndarray
            Depth array (same length as all arrays in ``data``).
        data : dict
            Feature name → depth-aligned array. All arrays must have
            the same length as ``depth``.
        weights : dict, optional
            Feature name → relative weight (positive float).
            Features not listed default to weight 1.
            Weights are applied by scaling each feature's values by
            sqrt(weight) before clustering, so a weight of 4 gives a
            feature twice the influence of weight 1.

        Returns
        -------
        FaciesResult
        """
        depth = np.asarray(depth, dtype=float)
        feature_names = list(data.keys())
        n = len(depth)

        X = np.column_stack([np.asarray(data[f], dtype=float) for f in feature_names])

        # Build weight vector
        w_vec = np.array([
            float(weights.get(f, 1.0)) if weights else 1.0
            for f in feature_names
        ])
        self._weights = w_vec

        # Mask rows with any NaN
        valid = ~np.any(np.isnan(X), axis=1)
        X_valid = X[valid]

        if X_valid.shape[0] < self.n_facies:
            raise ValueError(
                f"Only {X_valid.shape[0]} valid samples for {self.n_facies} facies. "
                "Reduce n_facies or clean NaN values."
            )

        # Standardize each feature, then apply sqrt(weight)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        X_weighted = X_scaled * np.sqrt(w_vec)

        kmeans = KMeans(
            n_clusters=self.n_facies,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        labels_valid = kmeans.fit_predict(X_weighted) + 1  # 1-based

        # Assign labels back (0 = unknown for NaN rows)
        labels = np.zeros(n, dtype=int)
        labels[valid] = labels_valid

        # Cluster centers back in original space
        centers_scaled = kmeans.cluster_centers_ / np.sqrt(w_vec)
        centers_orig = scaler.inverse_transform(centers_scaled)

        self._kmeans = kmeans
        self._scaler = scaler
        self._feature_names = feature_names

        return FaciesResult(
            depth=depth,
            facies=labels,
            n_facies=self.n_facies,
            feature_names=feature_names,
            weights=w_vec,
            cluster_centers=centers_orig,
            inertia=float(kmeans.inertia_),
        )

    def predict(
        self,
        data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Assign facies labels to new samples using the fitted model.

        Parameters
        ----------
        data : dict
            Same features as used in ``fit``, in the same order.

        Returns
        -------
        np.ndarray of int
            Facies labels (1-based; 0 = unknown / NaN input).
        """
        if self._kmeans is None:
            raise RuntimeError("Call fit() before predict()")

        X = np.column_stack([
            np.asarray(data[f], dtype=float) for f in self._feature_names
        ])
        valid = ~np.any(np.isnan(X), axis=1)
        n = X.shape[0]

        labels = np.zeros(n, dtype=int)
        if not np.any(valid):
            return labels
        X_scaled = self._scaler.transform(X[valid])
        X_weighted = X_scaled * np.sqrt(self._weights)
        labels[valid] = self._kmeans.predict(X_weighted) + 1

        return labels

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        facies_mnemonic: str = "FACIES",
        null_value: float = -999.25,
        **kwargs,
    ) -> "FaciesResult":
        """
        Load a pre-computed facies log from a LAS file.

        Depth is read via ``lasio``'s ``las.index`` property.  Use this when
        the facies were computed externally (e.g. by EpsLog) and you want to
        load them for analysis or plotting.

        Returns
        -------
        FaciesResult
            A minimal FaciesResult with depth and facies arrays.
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))
        depth = np.array(las.index, dtype=float)
        facies_raw = np.array(las[facies_mnemonic], dtype=float)
        # lasio already turns the NULL value into NaN
        facies_raw[(facies_raw == null_value) | ~np.isfinite(facies_raw)] = 0
        facies = facies_raw.astype(int)
        n_facies = int(np.max(facies[facies > 0])) if np.any(facies > 0) else 0

        return FaciesResult(
            depth=depth,
            facies=facies,
            n_facies=n_facies,
            feature_names=[],
            weights=np.array([]),
            cluster_centers=np.empty((0, 0)),
        )
