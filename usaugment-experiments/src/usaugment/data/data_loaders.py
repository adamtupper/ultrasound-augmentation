import os

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_data_loader(config, split, transform, shuffle):
    dataset = instantiate(
        config.dataset, root_dir=os.path.join(config.data_dir, config.version), split=split, transform=transform
    )

    if split == "train" and config.task == "classification":
        # Configure data loader with weighted random sampler using inverse weighted sampling
        labels = [item[config.dataset.label_key] for item in dataset.metadata]
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        class_weights = 1.0 / class_sample_count

        sample_weights = np.array([class_weights[t] for t in labels])
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=True if split == "train" else False,
        pin_memory=True,
        num_workers=config.workers,
        persistent_workers=True,
        prefetch_factor=config.prefetch_factor,
        sampler=sampler,
    )

    return data_loader
