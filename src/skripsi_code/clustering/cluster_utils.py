import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import numpy.typing as npt
from typing import Literal, List
from torch import Tensor
import wandb
from skripsi_code.clustering.cluster_methods import MiniK, Kmeans, GMM, Spectral, Agglomerative
from tqdm import tqdm


# @torch.compile
def mean_std_parameters(features: Tensor, epsilon: float = 1e-5) -> Tensor:
    size: int = features.size()

    # N: Batch size/samples, C: Number of dimentions
    assert len(size) == 2
    N = size[0]

    standard_deviation: Tensor = (
        (features.var(dim=1, unbiased=False) + epsilon).sqrt().view(N, 1)
    )
    mean: Tensor = features.mean(dim=1).view(N, 1)

    return standard_deviation, mean


def pseudolabel_reassignment(
    y_prev: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    # Asserting if both arrays has the same number of labels
    assert y_prev.size == y_pred.size

    # Kuhn-Munkers Algorithm, Weighted Biparte Matching
    Dimentions: int = max(y_prev.max(), y_pred.max()) + 1
    weights: npt.NDArray[np.int64] = np.zeros((Dimentions, Dimentions), dtype=np.int64)

    for i in range(y_prev.size):
        weights[y_prev[i], y_pred[i]] += 1

    row_index, column_index = linear_sum_assignment(weights.max() - weights)

    row_index: npt.NDArray[np.int64]
    column_index: npt.NDArray[np.int64]

    return column_index


def compute_statistics(
    dataloader: DataLoader,
    model: torch.nn.Module,
    N: int,
    device: Literal["cpu", "cuda"] = "cpu",
) -> npt.NDArray[np.float32]:
    # Put model in eval model
    model.eval()
    running_index = 0

    for batch, (X, *_) in enumerate(dataloader):
            input_tensor = X.double().to(device)
            original_shape = input_tensor.shape
            input_tensor = input_tensor.view(-1, original_shape[-1]) # Flatten to (N, features)
            batch_size = input_tensor.shape[0]
            features_extracted: Tensor = model.FeatureExtractorLayer.feature_extraction(
                input_tensor
            )

            for layer_idx, layer_features in enumerate(features_extracted):
                mean, standard_deviation = mean_std_parameters(layer_features)

                if layer_idx == 0:
                    auxilary = torch.cat((mean, standard_deviation), dim=1).numpy(
                        force=True
                    )
                else:
                    auxilary = np.concatenate(
                        (
                            auxilary,
                            torch.cat((mean, standard_deviation), dim=1).numpy(
                                force=True
                            ),
                        ),
                        axis=1,
                    )

            if batch == 0:
                features = np.zeros((N, auxilary.shape[1])).astype("float32")
            if batch < len(dataloader) - 1:
                features[running_index : running_index + batch_size] = auxilary.astype(
                    "float32"
                )
            else:
                features[running_index:] = auxilary.astype("float32")

            running_index += batch_size
            assert features.shape[1] == 2 * len(features_extracted)

    return features


def cluster_arrange(sample_lists: List) -> npt.NDArray[np.int64]:
    pseudolabels = []
    sample_idx = []

    for cluster, sample in enumerate(sample_lists):
        sample_idx.extend(sample)
        pseudolabels.extend([cluster] * len(sample))

    idx = np.argsort(sample_idx)
    return np.asarray(pseudolabels)[idx]


def pseudolabeling(
    dataset,
    model: torch.nn.Module,
    device: Literal["cpu", "cuda"],
    previous_cluster: npt.NDArray[np.int64],
    log_file: str,
    epoch: int,
    n_clusters: int,
    method: Literal["MiniK", "Kmeans", "GMM", "Spectral", "Agglomerative"] = "Kmeans",
    data_reduction=False,
    reduced_dimentions: int = 48,
    batch_size: int = 1,
    num_workers: int = 0,
    whitening: bool = False,
    L2norm: bool = False,
) -> npt.NDArray[np.int64]:
    cluster_object = globals()[method](
        n_clusters, reduced_dimentions, whitening, L2norm
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    features = compute_statistics(dataloader, model, dataset.length, device)

    cluster_object.cluster(data=features, data_reduction=data_reduction, verbose=False)
    cluster_arrangement = cluster_arrange(cluster_object.cluster_lists)

    class_nmi: float = normalized_mutual_info_score(
        cluster_arrangement,
        dataloader.dataset.cluster_label,
        average_method="geometric",
    )

    domain_nmi: float = normalized_mutual_info_score(
        cluster_arrangement, dataloader.dataset.domain_label, average_method="geometric"
    )

    previous_nmi: float = normalized_mutual_info_score(
        cluster_arrangement, previous_cluster, average_method="geometric"
    )

    log = "Epoch: {}, NMI against class labels: {:.3f}, domain labels: {:.3f}, previous assignment: {:.3f}".format(
        epoch, class_nmi, domain_nmi, previous_nmi
    )

    if log_file:
        # append log to file
        with open(log_file, "a") as f:
            f.write(log + "\n")

    if wandb.run is not None:
        wandb.log({
            "Clustering/NMI_Class": class_nmi,
            "Clustering/NMI_Domain": domain_nmi,
            "Clustering/NMI_Previous": previous_nmi
        }, step=epoch)

    # Get the mapping e.g: [1, 0, 3, 2] memiliki arti bahwa cluster 0 (index) akan di isi oleh data cluster 1 (value) dan seterusnya
    reassignment_mappings: List[int] = pseudolabel_reassignment(
        previous_cluster, cluster_arrangement
    )

    cluster_reassignment: List[List[int]] = [
        cluster_object.cluster_lists[mapping] for mapping in reassignment_mappings
    ]

    # mengubah list [[]] menjadi []
    return cluster_arrange(cluster_reassignment)


if __name__ == "__main__":
    tens = torch.randn(5, 6)

    print(tens)
    print(mean_std_parameters(tens))

    y_prev = np.random.randint(0, 4, (6,))
    y_post = np.random.randint(0, 4, (6,))

    print(pseudolabel_reassignment(y_prev, y_post))
