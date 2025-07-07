from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
from skripsi_code.utils.domain_dataset import MultiChunkDataset, MultiChunkParquet
from typing import List


def random_split_dataloader(
    dir_path: str,
    source_dir: List[str],
    target_dir: str,
    source_domain: List[str],
    target_domain: List[str],
    get_domain=False,
    get_cluster=False,
    batch_size=1,
    buffer_size=16,
    n_workers=0,
    chunk=True,
):
    source_data = MultiChunkParquet(
        dir_path,
        source_dir,
        domain=source_domain,
        get_domain=get_domain,
        get_cluster=get_cluster,
        buffer_size=buffer_size,
        chunk_mode=chunk,
    )
    target_data = MultiChunkParquet(
        dir_path,
        target_dir,
        domain=target_domain,
        get_domain=get_domain,
        get_cluster=get_cluster,
        buffer_size=buffer_size,
        chunk_mode=chunk,
    )

    source_train, source_val = random_split(source_data, [0.8, 0.2])
    source_train = deepcopy(source_train)

    print(
        "Train: {}, Val: {}, Test: {}".format(
            len(source_train), len(source_val), len(target_data)
        )
    )

    source_train = DataLoader(
        source_train, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True
    )
    source_val = DataLoader(
        source_val, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True
    )
    target_test = DataLoader(
        target_data, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True
    )
    return source_train, source_val, target_test

