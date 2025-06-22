import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS
from typing import Literal, List, Any
from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
import abc

__all__ = ["MiniK", "Kmeans", "GMM", "Spectral", "Agglomerative"]


def preprocessing_feature(
    data: npt.NDArray[np.float64],
    pca_dim: int = 48,
    whitening: bool = True,
    L2norm: bool = False,
    method: Literal["kpca", "isomap", "lle", "mds", "pca"] = "kpca",
) -> (Any, npt.NDArray[np.float64]):
    _, ndim = data.shape

    match method.lower():
        case "kpca":
            reduction = KernelPCA(n_components=pca_dim, kernel="rbf")
        case "isomap":
            reduction = Isomap(n_components=pca_dim)
        case "lle":
            reduction = LocallyLinearEmbedding(n_components=pca_dim)
        case "mds":
            reduction = MDS(n_components=pca_dim)
        case "pca":
            reduction = PCA(n_components=pca_dim, whiten=whitening)
        case _:
            reduction = PCA(n_components=pca_dim)

    data: npt.NDArray[np.float64] = reduction.fit_transform(data)

    if L2norm:
        row_sums = np.linalg.norm(data, axis=1)
        data = data / row_sums[:, np.newaxis]

    return reduction, data


class Clustering:
    def __init__(
        self,
        k: int = 3,
        pca_dim: int = 48,
        whitening: bool = False,
        L2norm: bool = False,
        method: Literal["kpca", "somap", "lle", "mds", "pca"] = "kpca",
    ):
        self.k = k
        self.pca_dim = pca_dim
        self.whitening = whitening
        self.L2norm = L2norm
        self.method = method
        self.cluster_lists: List = None
        self.reduction: Any = None
        self._clustering_model = None

    def cluster(
        self, data: npt.NDArray[np.float64], data_reduction=False, verbose: bool = False
    ) -> None:
        data = np.clip(data, -1e10, 1e10)
        if data_reduction:
            self.reduction, data_processed = preprocessing_feature(
                data=data,
                pca_dim=self.pca_dim,
                whitening=self.whitening,
                L2norm=self.L2norm,
                method=self.method,
            )
        else:
            data_processed = data

        # Avoiding replication
        self.cluster_lists = [[] for i in range(self.k)]

        if not self._clustering_model:
            self.fit_clustering(data_processed)

        clusters: npt.NDArray[int] = self.transform(data_processed)

        for i in range(len(data_processed)):
            # cluster labels are given based on their order, each sample is assign to its cluster based and its index to be appended
            self.cluster_lists[clusters[i]].append(i)

        return None

    @abc.abstractmethod
    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        pass

    def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[int]:
        return self._clustering_model.predict(data) if self._clustering_model else None


class MiniK(Clustering):
    def __init__(
        self,
        k: int = 3,
        pca_dim: int = 48,
        whitening: bool = False,
        L2norm: bool = False,
        method: Literal["kpca", "isomap", "lle", "mds", "pca"] = "kpca",
    ):
        super().__init__(k, pca_dim, whitening, L2norm, method)

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = MiniBatchKMeans(n_clusters=self.k, batch_size=2048)
        self._clustering_model.fit(data)


class Kmeans(Clustering):
    def __init__(
        self,
        k: int = 3,
        pca_dim: int = 48,
        whitening: bool = False,
        L2norm: bool = False,
        method: Literal["kpca", "isomap", "lle", "mds", "pca"] = "kpca",
    ):
        super().__init__(k, pca_dim, whitening, L2norm, method)

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = KMeans(n_clusters=self.k)
        self._clustering_model.fit(data)


class GMM(Clustering):
    def __init__(
        self,
        k: int = 3,
        pca_dim: int = 48,
        whitening: bool = False,
        L2norm: bool = False,
        method: Literal["kpca", "isomap", "lle", "mds", "pca"] = "kpca",
    ):
        super().__init__(k, pca_dim, whitening, L2norm, method)

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = GaussianMixture(n_clusters=self.k)
        self._clustering_model.fit(data)


class Spectral(Clustering):
    def __init__(
        self,
        k: int = 3,
        pca_dim: int = 48,
        whitening: bool = False,
        L2norm: bool = False,
        method: Literal["kpca", "isomap", "lle", "mds", "pca"] = "kpca",
    ):
        super().__init__(k, pca_dim, whitening, L2norm, method)

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = SpectralClustering(n_clusters=self.k)
        self._clustering_model.fit(data)


class Agglomerative(Clustering):
    def __init__(
        self,
        k: int = 3,
        pca_dim: int = 48,
        whitening: bool = False,
        L2norm: bool = False,
        method: Literal["kpca", "isomap", "lle", "mds", "pca"] = "kpca",
    ):
        super().__init__(k, pca_dim, whitening, L2norm, method)

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = AgglomerativeClustering(n_clusters=self.k)
        self._clustering_model.fit(data)
