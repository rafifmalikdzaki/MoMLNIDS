from sympy.codegen.ast import float32
from torch.utils.data import Dataset
import torch
from typing import List, Literal
import numpy as np
import pandas as pd
import shutil
import subprocess
from pathlib import Path
import glob
import polars as pl
from tqdm import tqdm


class MultiChunkParquet(Dataset):
    def __init__(
        self,
        dir_path: str,
        directories: List[str],
        domain: str,
        get_domain=False,
        get_cluster=False,
        chunk_mode=True,
        chunk_size=100_000,
        buffer_size=32,
    ):
        self.domain = domain
        self.domain_label = None
        self.cluster_label = None
        self.chunk_mode = chunk_mode

        self.get_domain = get_domain
        self.get_cluster = get_cluster

        self.PATH = dir_path
        self.parquet_files = list()
        self.chunk_size = chunk_size
        self.buffer = {}
        self.buffer_size = buffer_size

        self.chunk_count = 0
        self.length = 0

        self.directories = directories

        self.__load_data()
        self.__load_chunk()

    def __len__(self):
        if self.chunk_mode:
            return self.chunk_count
        else:
            return self.length

    def __getitem__(self, idx):
        if self.chunk_mode:
            sample = pl.read_parquet(self.parquet_files[idx])

            _features = sample.select(pl.nth(range(4, 43))).with_columns(
                pl.col("*").cast(pl.Float32)
            )
            _label = sample.select(pl.nth(43)).cast(pl.Int64)

            features = torch.tensor(_features.to_numpy())
            features = torch.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5)
            label = torch.tensor(_label.to_numpy()).squeeze()
            output = [features, label]

            start_index = idx * self.chunk_size
            end_index = start_index + self.chunk_size
            data_range = range(start_index, end_index)

            if self.get_domain:
                domain = np.copy(self.domain_label[data_range])
                domain = np.int64(domain)
                output.append(domain)

            if self.get_cluster:
                cluster = np.copy(self.cluster_label[data_range])
                cluster = np.int64(cluster)
                output.append(cluster)

        else:
            chunk_index = idx // self.chunk_size
            sample_index = idx % self.chunk_size

            chunk = self.__load_chunk(chunk_index)
            sample = chunk[sample_index]

            _features = sample.select(pl.nth(range(4, 43))).with_columns(
                pl.col("*").cast(pl.Float32)
            )
            _label = sample.select(pl.nth(43)).cast(pl.Int64)

            features = torch.tensor(_features.to_numpy().astype(np.float32)).squeeze()
            features = torch.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5)
            label = torch.tensor(_label.to_numpy().astype(np.int64)).squeeze()
            output = [features, label]

            if self.get_domain:
                domain = np.copy(self.domain_label[sample_index])
                domain = np.int64(domain)
                output.append(domain)

            if self.get_cluster:
                cluster = np.copy(self.cluster_label[sample_index])
                cluster = np.int64(cluster)
                output.append(cluster)

        return output

    def reload_buffer(self):
        if not self.chunk_mode:
            self.buffer.clear()
            self.__load_chunk()

    def set_cluster(self, cluster_list) -> None:
        if len(cluster_list) != self.length:
            raise ValueError(
                "The length of cluster_list must to be same as self.features"
            )
        else:
            self.cluster_label = cluster_list

    def set_domain(self, domain_list) -> None:
        if len(domain_list) != self.length:
            raise ValueError(
                "The length of domain_list must to be same as self.features"
            )
        else:
            self.domain_label = domain_list

    def __load_data(self):
        if not isinstance(self.directories, List):
            self.directories = [self.directories]

        print(f"Data directories: {self.directories}")

        self.domain_label = np.zeros(0)

        for idx, directory in enumerate(self.directories):
            domain_samples = 0
            files = glob.glob(f"{self.PATH}/{directory}/*.parquet")
            self.parquet_files.extend(files)
            for file in tqdm(files, desc=f"Loading {directory}"):
                data = pl.scan_parquet(file)
                data_length = data.select(pl.len()).collect().item()
                self.length += data_length
                domain_samples += data_length
            self.domain_label = np.append(
                self.domain_label, np.ones(domain_samples) * idx
            )

        self.parquet_files.sort()
        self.chunk_count = len(self.parquet_files)
        self.cluster_label = np.zeros(len(self.domain_label), dtype=np.int64)

    def __load_chunk(self, chunk_index=None):
        # Initial buffer population if buffer is empty
        if not self.buffer:
            for chunk_idx in range(
                min(self.buffer_size, self.chunk_count)
            ):  # Ensure we don't exceed chunk_count
                chunk = pl.read_parquet(self.parquet_files[chunk_idx])
                self.buffer[chunk_idx] = chunk
            return None

        # If the requested chunk is not in the buffer, load it and add it to the buffer
        if chunk_index not in self.buffer:
            chunk = pl.read_parquet(self.parquet_files[chunk_index])
            self.buffer[chunk_index] = chunk  # Add it to the buffer

        # Shift buffer contents and load new chunks as needed
        if chunk_index + 1 not in self.buffer:
            for idx in range(1, self.buffer_size):
                next_chunk_index = chunk_index + idx
                if next_chunk_index < self.chunk_count:  # Boundary check
                    chunk = pl.read_parquet(self.parquet_files[next_chunk_index])
                    self.buffer[next_chunk_index] = chunk  # Add next chunk to buffer
                # Remove old chunks that are no longer in the range of current buffer
                previous_chunk_index = chunk_index - idx
                if previous_chunk_index in self.buffer:
                    del self.buffer[previous_chunk_index]

        return self.buffer[chunk_index]


# Update the get item and loading dataset to load the domain and cluster
class MultiChunkDataset(Dataset):
    def __init__(
        self,
        dir_path: str,
        directories: List[str],
        domain: str,
        get_domain=False,
        get_cluster=False,
        chunk_mode=False,
        chunk_size=50_000,
        buffer_size=10,
    ):
        self.domain = domain
        self.domain_label = None
        self.cluster_label = None
        self.chunk_mode = chunk_mode

        self.get_domain = get_domain
        self.get_cluster = get_cluster

        self.PATH = dir_path
        self.csv_files = list()
        self.chunk_size = chunk_size
        self.buffer = {}
        self.buffer_size = buffer_size

        self.chunk_count = 0
        self.length = 0
        self.chunk_length = list()

        self.directories = directories

        self.__load_data()
        self.__load_chunk()

    def __len__(self):
        if self.chunk_mode:
            return self.chunk_count
        else:
            return self.length

    def __getitem__(self, idx):
        if self.chunk_mode:
            sample = self.__load_chunk(idx)

            _features = sample.select(pl.nth(range(4, 43))).with_columns(
                pl.col("*").cast(pl.Float32)
            )
            _label = sample.select(pl.nth(43)).cast(pl.Int32)

            features = torch.tensor(_features.to_numpy())
            label = torch.tensor(_label.to_numpy()).squeeze()
        else:
            chunk_index = idx // self.chunk_size
            sample_index = idx % self.chunk_size

            chunk = self.__load_chunk(chunk_index)
            sample = chunk[sample_index]

            _features = sample.select(pl.nth(range(4, 43))).with_columns(
                pl.col("*").cast(pl.Float32)
            )
            _label = sample.select(pl.nth(43)).cast(pl.Int32)

            features = torch.tensor(_features.to_numpy())
            label = torch.tensor(_label.item())

        output = (features, label)

        # if self.domain_label:
        #     domain = np.copy(self.domain_label[idx]).item()
        #     output.append(domain)

        return features, label

    def reload_buffer(self):
        self.buffer.clear()
        self.__load_chunk()

    def set_cluster(self, cluster_list) -> None:
        if len(cluster_list) != self.length:
            raise ValueError(
                "The length of cluster_list must to be same as self.features"
            )
        else:
            self.cluster_label = cluster_list

    def set_domain(self, domain_list) -> None:
        if len(domain_list) != self.length:
            raise ValueError(
                "The length of domain_list must to be same as self.features"
            )
        else:
            self.domain_label = domain_list

    def __load_data(self):
        if not isinstance(self.directories, List):
            self.directories = [self.directories]

        print(f"Data directories: {self.directories}")

        for directory in self.directories:
            self.csv_files.extend(glob.glob(f"{self.PATH}/{directory}/*.csv"))

        self.csv_files.sort()
        self.chunk_count = len(self.csv_files)

        for file in tqdm(self.csv_files):
            data = pl.scan_csv(file)
            data_length = data.select(pl.len()).collect().item()
            self.chunk_length.append(data_length)
            self.length += data_length

        self.domain_label = np.zeros(0)
        self.cluster_label = np.zeros(0)

    def __load_chunk(self, chunk_index=None):
        # Initial buffer population if buffer is empty
        if not self.buffer:
            for chunk_idx in range(
                min(self.buffer_size, self.chunk_count)
            ):  # Ensure we don't exceed chunk_count
                chunk = pl.read_csv(self.csv_files[chunk_idx])
                self.buffer[chunk_idx] = chunk
            return None

        # If the requested chunk is not in the buffer, load it and add it to the buffer
        if chunk_index not in self.buffer:
            chunk = pl.read_csv(self.csv_files[chunk_index])
            self.buffer[chunk_index] = chunk  # Add it to the buffer

        # Shift buffer contents and load new chunks as needed
        if chunk_index + 1 not in self.buffer:
            for idx in range(1, self.buffer_size):
                next_chunk_index = chunk_index + idx
                if next_chunk_index < self.chunk_count:  # Boundary check
                    chunk = pl.read_csv(self.csv_files[next_chunk_index])
                    self.buffer[next_chunk_index] = chunk  # Add next chunk to buffer
                # Remove old chunks that are no longer in the range of current buffer
                previous_chunk_index = chunk_index - idx
                if previous_chunk_index in self.buffer:
                    del self.buffer[previous_chunk_index]

        return self.buffer[chunk_index]


class Fraction_Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_files: List[str],
        split: Literal["train", "test"] = "train",
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.data_files = data_files
        self.split = split
        self.domain_label = None
        self.cluster_label = None
        self.length = None

        self.__load_dataset(self.split)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = next(self.__data_generator())
        features = torch.as_tensor(row[4:-2], dtype=torch.float32)
        label = torch.as_tensor(row[-2])

        if self.domain_label:
            domain = torch.as_tensor(self.domain_label[idx])
        elif self.cluster_label:
            domain = torch.as_tensor(self.cluster_label[idx])
        else:
            raise Exception("No domain labels present!")

        return features, label, domain

    def __load_dataset(self, split: str) -> None:
        if split.lower() == "train":
            self.filename = self.data_path / "Train.csv"
            COMB_FILE = [self.data_path / data for data in self.data_files]
            with open(self.filename, "w", newline="") as outfile:
                for file_idx, input_file in enumerate(COMB_FILE):
                    with open(input_file, "r") as infile:
                        if file_idx > 0:
                            next(infile)
                        shutil.copyfileobj(infile, outfile)

        elif split.lower() == "test":
            self.filename = self.data_path / "Test.csv"

        else:
            raise ValueError("The spliting isnt valid")

        self.length = (
            int(
                subprocess.check_output(f"wc -l {self.filename}", shell=True).split()[0]
            )
            - 1
        )

    def set_cluster(self, cluster_list) -> None:
        if len(cluster_list) != len(self.features):
            raise ValueError(
                "The length of cluster_list must to be same as self.features"
            )
        else:
            self.clusters_label = cluster_list

    def set_domain(self, domain_list) -> None:
        if len(domain_list) != len(self.features):
            raise ValueError(
                "The length of domain_list must to be same as self.features"
            )
        else:
            self.domain_label = domain_list


class Whole_Dataset(Dataset):
    def __init__(
        self, data_path: str, data_files: List[str], domain: List[int], seed: int = 42
    ):
        self.data_path = data_path
        self.data_files = data_files
        self.domain_label = None
        self.cluster_label = None
        self.seed = seed

        data = (
            pd.concat(
                [pd.read_csv(data_path + file) for file in data_files],
                ignore_index=True,
            )
            .iloc[:, 4:]
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
        )

        self.binary_label = data["Label"].to_numpy()
        self.multi_label = data["Attack"].to_numpy()

        self.features = data.drop(columns=["Label", "Attack"], axis=1).to_numpy()

        del data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = np.float64(self.features.iloc[idx])
        label = np.int64(self.binary_label[idx])
        domain = np.int64(self.domain_label[idx])
        return features, label, domain

    def set_cluster(self, cluster_list) -> None:
        if len(cluster_list) != len(self.features):
            raise ValueError(
                "The length of cluster_list must to be same as self.features"
            )
        else:
            self.clusters_label = cluster_list

    def set_domain(self, domain_list) -> None:
        if len(domain_list) != len(self.features):
            raise ValueError(
                "The length of domain_list must to be same as self.features"
            )
        else:
            self.domain_label = domain_list
