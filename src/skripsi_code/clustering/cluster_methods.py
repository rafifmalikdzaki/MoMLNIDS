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
import time

# Import optional dependencies with fallbacks
try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
        if hasattr(self._clustering_model, "predict"):
            return self._clustering_model.predict(data)
        elif hasattr(self._clustering_model, "fit_predict"):
            return self._clustering_model.fit_predict(data)
        else:
            return None


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
        self._clustering_model = GaussianMixture(n_components=self.k)
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
        self._data = None

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = SpectralClustering(n_clusters=self.k)
        # SpectralClustering doesn't have a separate fit method, so we store the data
        self._data = data

    def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[int]:
        # For SpectralClustering, we need to use fit_predict as it doesn't have predict
        return self._clustering_model.fit_predict(data)


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
        self._data = None

    def fit_clustering(
        self,
        data: npt.NDArray[np.float64],
    ) -> None:
        self._clustering_model = AgglomerativeClustering(n_clusters=self.k)
        # AgglomerativeClustering doesn't have a separate fit method, so we store the data
        self._data = data

    def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[int]:
        # For AgglomerativeClustering, we need to use fit_predict as it doesn't have predict
        return self._clustering_model.fit_predict(data)


def demo_clustering_methods():
    """Demonstrate all clustering methods with sample data."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit("🔬 Clustering Methods Demo", style="bold blue"))
    else:
        print("🔬 Clustering Methods Demo")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_clusters = 3

    # Create sample data with some structure
    data = np.random.randn(n_samples, n_features)
    for i in range(n_clusters):
        cluster_center = np.random.randn(n_features) * 3
        cluster_data = (
            np.random.randn(n_samples // n_clusters, n_features) + cluster_center
        )
        data[i * (n_samples // n_clusters) : (i + 1) * (n_samples // n_clusters)] = (
            cluster_data
        )

    print(
        f"📊 Generated sample data: {data.shape[0]} samples, {data.shape[1]} features"
    )

    # Test all clustering methods
    clustering_methods = {
        "K-Means": Kmeans,
        "Mini-Batch K-Means": MiniK,
        "Gaussian Mixture Model": GMM,
        "Spectral Clustering": Spectral,
        "Agglomerative Clustering": Agglomerative,
    }

    if RICH_AVAILABLE:
        results_table = Table(title="Clustering Results")
        results_table.add_column("Method", style="cyan", no_wrap=True)
        results_table.add_column("Time (s)", style="magenta")
        results_table.add_column("Clusters Found", style="green")
        results_table.add_column("Status", style="yellow")

        methods_to_test = track(
            clustering_methods.items(), description="Testing clustering methods..."
        )
    else:
        methods_to_test = clustering_methods.items()

    for method_name, method_class in methods_to_test:
        try:
            start_time = time.time()

            # Initialize clustering method
            clusterer = method_class(k=n_clusters, pca_dim=8)

            # Perform clustering
            clusterer.cluster(data, data_reduction=True, verbose=False)

            end_time = time.time()
            execution_time = end_time - start_time

            # Check results
            cluster_counts = [
                len(cluster_list) for cluster_list in clusterer.cluster_lists
            ]
            non_empty_clusters = sum(1 for count in cluster_counts if count > 0)

            status = "✅ Success"

            if RICH_AVAILABLE:
                results_table.add_row(
                    method_name,
                    f"{execution_time:.3f}",
                    str(non_empty_clusters),
                    status,
                )

            print(
                f"  └─ {method_name}: {non_empty_clusters} clusters, largest: {max(cluster_counts)} samples"
            )

        except Exception as e:
            if RICH_AVAILABLE:
                results_table.add_row(
                    method_name, "N/A", "N/A", f"❌ Error: {str(e)[:30]}..."
                )
            else:
                print(f"  └─ {method_name}: ❌ Error: {str(e)[:50]}...")

    if RICH_AVAILABLE:
        console.print(results_table)

    # Demonstrate preprocessing methods
    if RICH_AVAILABLE:
        console.print("\n")
        console.print(Panel.fit("🔧 Preprocessing Methods Demo", style="bold green"))
    else:
        print("\n🔧 Preprocessing Methods Demo")

    preprocessing_methods = ["pca", "kpca", "isomap", "lle", "mds"]

    if RICH_AVAILABLE:
        preprocessing_table = Table(title="Dimensionality Reduction Results")
        preprocessing_table.add_column("Method", style="cyan")
        preprocessing_table.add_column("Original Dims", style="blue")
        preprocessing_table.add_column("Reduced Dims", style="green")
        preprocessing_table.add_column("Time (s)", style="magenta")
        preprocessing_table.add_column("Status", style="yellow")

        methods_to_test = track(
            preprocessing_methods, description="Testing preprocessing methods..."
        )
    else:
        methods_to_test = preprocessing_methods

    for method in methods_to_test:
        try:
            start_time = time.time()
            reduction_model, processed_data = preprocessing_feature(
                data, pca_dim=5, method=method
            )
            end_time = time.time()
            execution_time = end_time - start_time

            if RICH_AVAILABLE:
                preprocessing_table.add_row(
                    method.upper(),
                    str(data.shape[1]),
                    str(processed_data.shape[1]),
                    f"{execution_time:.3f}",
                    "✅ Success",
                )
            else:
                print(
                    f"  └─ {method.upper()}: {data.shape[1]} -> {processed_data.shape[1]} dims in {execution_time:.3f}s"
                )

        except Exception as e:
            if RICH_AVAILABLE:
                preprocessing_table.add_row(
                    method.upper(),
                    str(data.shape[1]),
                    "N/A",
                    "N/A",
                    f"❌ Error: {str(e)[:20]}...",
                )
            else:
                print(f"  └─ {method.upper()}: ❌ Error: {str(e)[:50]}...")

    if RICH_AVAILABLE:
        console.print(preprocessing_table)
        console.print("\n✨ Demo completed! All clustering methods have been tested.")
    else:
        print("\n✨ Demo completed! All clustering methods have been tested.")


# Only add click decorator if click is available
if CLICK_AVAILABLE:

    @click.command()
    @click.option("--demo", is_flag=True, help="Run clustering methods demonstration")
    @click.option(
        "--method",
        type=click.Choice(["kmeans", "minik", "gmm", "spectral", "agglomerative"]),
        help="Test specific clustering method",
    )
    @click.option("--n-samples", default=1000, help="Number of samples for test data")
    @click.option("--n-clusters", default=3, help="Number of clusters")
    @click.option("--n-features", default=10, help="Number of features")
    def main(demo, method, n_samples, n_clusters, n_features):
        """
        Test and demonstrate clustering methods functionality.

        This script provides comprehensive testing of all clustering algorithms
        and preprocessing methods available in the module.
        """
        if RICH_AVAILABLE:
            console = Console()

        if demo:
            demo_clustering_methods()
        elif method:
            if RICH_AVAILABLE:
                console.print(
                    Panel.fit(
                        f"🔬 Testing {method.upper()} Clustering", style="bold blue"
                    )
                )
            else:
                print(f"🔬 Testing {method.upper()} Clustering")

            # Generate test data
            np.random.seed(42)
            data = np.random.randn(n_samples, n_features)

            # Select method
            method_map = {
                "kmeans": Kmeans,
                "minik": MiniK,
                "gmm": GMM,
                "spectral": Spectral,
                "agglomerative": Agglomerative,
            }

            clusterer = method_map[method](k=n_clusters)

            start_time = time.time()
            clusterer.cluster(data, data_reduction=True)
            end_time = time.time()

            if RICH_AVAILABLE:
                console.print(
                    f"✅ Clustering completed in {end_time - start_time:.3f} seconds"
                )
            else:
                print(f"✅ Clustering completed in {end_time - start_time:.3f} seconds")

            # Display results
            cluster_counts = [
                len(cluster_list) for cluster_list in clusterer.cluster_lists
            ]
            for i, count in enumerate(cluster_counts):
                if RICH_AVAILABLE:
                    console.print(f"  Cluster {i}: {count} samples")
                else:
                    print(f"  Cluster {i}: {count} samples")
        else:
            if RICH_AVAILABLE:
                console.print(
                    "Use --demo to run full demonstration or --method to test specific clustering method"
                )
                console.print("Use --help for more options")
            else:
                print(
                    "Use --demo to run full demonstration or --method to test specific clustering method"
                )
                print("Use --help for more options")
else:

    def main():
        """
        Test and demonstrate clustering methods functionality.
        """
        print("Clustering Methods Module")
        print("Available functions:")
        print("- demo_clustering_methods(): Run comprehensive clustering demo")
        print("- Clustering classes: Kmeans, MiniK, GMM, Spectral, Agglomerative")

        # Run demo by default when click is not available
        demo_clustering_methods()
