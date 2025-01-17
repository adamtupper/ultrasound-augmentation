from typing import Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from omegaconf import DictConfig

from .trivial_augment import configure_trivial_augment


def get_data_loader_transforms(config: DictConfig) -> Tuple[A.Compose, A.Compose]:
    """Get the transforms for the train and test dataloaders."""

    if config.augmentation._target_ == "albumentations.Compose":
        # Using TrivialAugment, filter augmentations in the top N
        for i in range(len(config.augmentation.transforms)):
            config.augmentation.transforms[i]["transforms"] = config.augmentation.transforms[i]["transforms"][
                : config.top_n_augmentations
            ]

    train_transform_list = [
        A.LongestMaxSize(max_size=256, interpolation=1),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=0.0, std=1.0),  # Normalize pixel values to [0, 1]
        instantiate(config.augmentation),
        # If random crop is used, this will have no effect. However, if it isn't used the image will need to be resized
        # to 224 x 224 px before being passed to the model
        A.LongestMaxSize(max_size=224, interpolation=1),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(transpose_mask=True),
    ]

    test_transform_list = [
        A.LongestMaxSize(max_size=224, interpolation=1),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=0.0, std=1.0),  # Normalize pixel values to [0, 1]
        ToTensorV2(transpose_mask=True),
    ]

    train_transform = A.Compose(train_transform_list, additional_targets={"scan_mask": "mask"})
    test_transform = A.Compose(test_transform_list, additional_targets={"scan_mask": "mask"})

    return train_transform, test_transform


def get_test_transform(config: DictConfig) -> A.Compose:
    """Get the transforms for the test dataloader."""

    test_transform_list = [
        A.LongestMaxSize(max_size=224, interpolation=1),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=0.0, std=1.0),  # Normalize pixel values to [0, 1]
        ToTensorV2(transpose_mask=True),
    ]

    test_transform = A.Compose(test_transform_list, additional_targets={"scan_mask": "mask"})

    return test_transform


def get_trivial_augment_transform(config: DictConfig) -> A.Compose:
    """Get the TrivialAugment transform."""

    train_transform_list = [
        A.LongestMaxSize(max_size=256, interpolation=1),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=0.0, std=1.0),  # Normalize pixel values to [0, 1]
        configure_trivial_augment(config),
        # If random crop is used, this will have no effect. However, if it isn't used the image will need to be resized
        # to 224 x 224 px before being passed to the model
        A.LongestMaxSize(max_size=224, interpolation=1),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(transpose_mask=True),
    ]

    train_transform = A.Compose(train_transform_list, additional_targets={"scan_mask": "mask"})

    return train_transform
