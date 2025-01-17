# %%
# Setup
import math

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from albumentations.pytorch import ToTensorV2
from usaugment.data import ClassificationDataset, SegmentationDataset

GBCU_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/gbcu_v2"
MMOTU_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/mmotu_v2"
STANFORD_THYROID_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/stanford_thyroid_v4"
CAMUS_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/camus_v4"
NFLD_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/fatty_liver_v3"
AUL_LIVER_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/aul_liver_v5"
AUL_MASS_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/aul_mass_v5"
BUTTERFLY_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/butterfly_v4"
OPEN_KIDNEY_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/open_kidney_v3"
POCUS_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/pocus_v4"
PSFHS_ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/psfhs_v2"


# %%
# Classification tasks
def get_combined_df(train_df, val_df, test_df):
    train_df["split"] = "Train"
    val_df["split"] = "Val"
    test_df["split"] = "Test"
    return pd.concat([train_df, val_df, test_df])


sns.set_theme(style="whitegrid", context="paper")

datasets = [
    ("GBCU", ClassificationDataset, GBCU_ROOT_DIR, ["Normal", "Benign", "Malignant"], "label"),
    (
        "MMOTU",
        ClassificationDataset,
        MMOTU_ROOT_DIR,
        [
            "Chocolate cyst",
            "Serious cystadenoma",
            "Teratoma",
            "Theca cell tumor",
            "Simple cyst",
            "Normal ovary",
            "Mucinous cystadenoma",
            "High grade serous",
        ],
        "label",
    ),
    ("Fatty Liver", ClassificationDataset, NFLD_ROOT_DIR, ["Normal", "NFLD"], "label"),
    ("AUL Mass", ClassificationDataset, AUL_MASS_ROOT_DIR, ["Normal", "Benign", "Malignant"], "label"),
    (
        "Butterfly",
        ClassificationDataset,
        BUTTERFLY_ROOT_DIR,
        ["Carotid", "2 channel", "Lungs", "IVC", "4 channel", "Bladder", "Thyroid", "Plax", "Morisons Pouch"],
        "label",
    ),
    ("CAMUS", ClassificationDataset, CAMUS_ROOT_DIR, ["Poor", "Medium", "Good"], "label"),
    ("POCUS", ClassificationDataset, POCUS_ROOT_DIR, ["Regular", "Pneumonia", "COVID-19"], "label"),
]

fig, axes = plt.subplots(nrows=2, ncols=math.ceil(len(datasets) / 2), figsize=(12, 6))

for i, ax in zip(range(len(datasets)), axes.flat):
    title, factory, root_dir, tick_labels, label_key = datasets[i]
    ax.set_title(datasets[i])

    train_df = pd.DataFrame.from_records(factory(root_dir, "train", transform=None, label_key=label_key).metadata)
    val_df = pd.DataFrame.from_records(factory(root_dir, "validation", transform=None, label_key=label_key).metadata)
    test_df = pd.DataFrame.from_records(factory(root_dir, "test", transform=None, label_key=label_key).metadata)
    combined_df = get_combined_df(train_df, val_df, test_df)

    sns.countplot(
        ax=ax,
        data=combined_df,
        x=label_key,
        # hue="split",
    )
    # ax.get_legend().remove()

    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel("Examples", fontsize=8)
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

axes[-1, -1].axis("off")

plt.tight_layout()
plt.savefig("../outputs/classification_tasks.pdf")
plt.show()


# %%
# Segmentation tasks
transform = A.Compose(
    [
        A.LongestMaxSize(max_size=224, interpolation=1),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(transpose_mask=True),
    ],
    additional_targets={"scan_mask": "mask"},
)

datasets = [
    ("MMOTU", SegmentationDataset(MMOTU_ROOT_DIR, "train", transform=transform, mask_key="tumor_mask_binary"), 0),
    (
        "Stanford Thyroid",
        SegmentationDataset(STANFORD_THYROID_ROOT_DIR, "train", transform=transform, mask_key="tumor_mask"),
        0,
    ),
    (
        "Open Kidney",
        SegmentationDataset(OPEN_KIDNEY_ROOT_DIR, "train", transform=transform, mask_key="capsule_mask"),
        0,
    ),
    ("CAMUS", SegmentationDataset(CAMUS_ROOT_DIR, "train", transform=transform), 0),
    ("AUL (liver)", SegmentationDataset(AUL_LIVER_ROOT_DIR, "train", transform=transform, mask_key="liver_mask"), 0),
    ("AUL (mass)", SegmentationDataset(AUL_MASS_ROOT_DIR, "train", transform=transform, mask_key="mass_mask"), 0),
    ("PSFHS", SegmentationDataset(PSFHS_ROOT_DIR, "train", transform=transform, mask_key="psfh_mask"), 0),
]

fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=(11, 3.5), sharey=True)
for i, rows in enumerate(axes):
    for j, ax in enumerate(rows):
        title, dataset, index = datasets[j]

        if i == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            if dataset is not None:
                ax.imshow(dataset[index][0].permute(1, 2, 0), cmap="gray")
            else:
                ax.annotate(
                    "Placeholder",
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            if dataset is not None:
                mask = dataset[index][1].permute(1, 2, 0)
                ax.imshow(mask, cmap="gray")
            else:
                ax.annotate(
                    "Placeholder",
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )

plt.tight_layout()
plt.savefig("../outputs/segmentation_tasks.pdf")
plt.show()

# %%
# Scan masks
transform = A.Compose(
    [
        A.LongestMaxSize(max_size=224, interpolation=1),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2(transpose_mask=True),
    ],
    additional_targets={"scan_mask": "mask"},
)

datasets = [
    ("MMOTU", SegmentationDataset(MMOTU_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 3),
    (
        "Stanford Thyroid",
        SegmentationDataset(STANFORD_THYROID_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"),
        0,
    ),
    (
        "Open Kidney",
        SegmentationDataset(OPEN_KIDNEY_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"),
        0,
    ),
    ("AUL", SegmentationDataset(AUL_LIVER_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 0),
    ("Butterfly", SegmentationDataset(BUTTERFLY_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 22000),
    ("CAMUS", SegmentationDataset(CAMUS_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 0),
    ("Fatty Liver", SegmentationDataset(NFLD_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 0),
    ("GBCU", SegmentationDataset(GBCU_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 0),
    ("POCUS", SegmentationDataset(POCUS_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 120),
    ("PSFHS", SegmentationDataset(PSFHS_ROOT_DIR, "train", transform=transform, mask_key="scan_mask"), 0),
]

fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=(11, 2.5), sharey=True)
for i, rows in enumerate(axes):
    for j, ax in enumerate(rows):
        title, dataset, index = datasets[j]

        if i == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            if dataset is not None:
                ax.imshow(dataset[index][0].permute(1, 2, 0), cmap="gray")
            else:
                ax.annotate(
                    "Placeholder",
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            if dataset is not None:
                mask = dataset[index][1].permute(1, 2, 0)
                ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
            else:
                ax.annotate(
                    "Placeholder",
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )

plt.tight_layout()
plt.savefig("../outputs/scan_masks.pdf")
plt.show()

# %%
