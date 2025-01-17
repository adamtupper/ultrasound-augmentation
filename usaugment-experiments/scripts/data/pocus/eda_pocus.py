"""The POCUS COVID-19 lung ultrasound dataset was published alongside the paper "Accelerating Detection of Lung
Pathologies with Explainable Ultrasound Image Analysis" (Born et al., 2021).

Dataset: https://github.com/jannisborn/covid19_ultrasound/
Paper: http://dx.doi.org/10.3390/app11020672
"""

# %%
# Setup
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import skimage.io as io
from skimage.util import compare_images

ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/pocus_v4/"

# %%
# Check that all images are 3-channel
image_paths = glob.glob(os.path.join(ROOT_DIR, "*", "*.png"))
for image_path in image_paths:
    image = io.imread(image_path)
    assert (image.ndim == 3) and (image.shape[-1] == 3), f"Image {image_path} has {image.ndim} dimensions"


# %%
# Are the pixel values in each channel of each image identical?
image_paths = glob.glob(os.path.join(ROOT_DIR, "*", "*.png"))
affected_images = []
for image_path in image_paths:
    image = io.imread(image_path)

    channels_are_identical = np.all(image[:, :, :1] == image)
    if not channels_are_identical:
        affected_images.append(image_path)

print(f"All channels are identical in {len(affected_images)} out of {len(image_paths)} images")

# %%
# Visualize each affected image and each channel separately
fig, axes = plt.subplots(4, 10, figsize=(20, 8))
for i, image_path in enumerate(affected_images[:10]):
    image = io.imread(image_path)

    axes[0, i].imshow(image)
    axes[0, i].set_title("RGB")
    axes[0, i].axis("off")

    for j in range(3):
        axes[j + 1, i].imshow(image[:, :, j], cmap="gray")
        axes[j + 1, i].set_title(f"Channel {j}")
        axes[j + 1, i].axis("off")

plt.tight_layout()
plt.show()


# %%
# Visualize the difference between each pair of channels [(0, 1), (0, 2), (1, 2)] for each of the first 10 affected
# images
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i, image_path in enumerate(affected_images[:10]):
    image = io.imread(image_path)

    for j, (channel1, channel2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        diff = compare_images(image[:, :, channel1], image[:, :, channel2])

        axes[j, i].imshow(diff, cmap="gray")
        axes[j, i].set_title(f"Channels {channel1} and {channel2}")
        axes[j, i].axis("off")

plt.tight_layout()
plt.show()

# %%
# Load the metadata
train_examples = pd.read_json(os.path.join(ROOT_DIR, "train.json"))
train_examples["split"] = "train"
validation_examples = pd.read_json(os.path.join(ROOT_DIR, "validation.json"))
validation_examples["split"] = "validation"
test_examples = pd.read_json(os.path.join(ROOT_DIR, "test.json"))
test_examples["split"] = "test"
examples = pd.concat([train_examples, validation_examples, test_examples])


def load_image_statistics(row):
    image = io.imread(os.path.join(ROOT_DIR, row["image"]))

    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "min_px_value": image.min(),
        "max_px_value": image.max(),
    }


examples[["width", "height", "min_px_value", "max_px_value"]] = examples.apply(
    load_image_statistics, axis=1, result_type="expand"
)

examples.head()

# %%
# Plot the class distribution of the each split

sns.catplot(data=examples, x="pathology", hue="split", kind="count", height=5, aspect=2)
plt.title("Class Distribution of each Dataset Split")
plt.xlabel("Pathology")
plt.ylabel("Count")
plt.show()

# %%
# Plot the size of each image as a point on a scatter plot, coloured by class
fig, ax = plt.subplots()

symbols = ["o", "s", "^"]
for i, (pathology, group) in enumerate(examples.groupby("pathology")):
    ax.scatter(
        group["width"], group["height"], label=pathology, marker=symbols[i], color=sns.color_palette()[i], alpha=0.5
    )

ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
ax.set_aspect("equal")
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1200)
ax.legend()

plt.title("Image dimensions by class")
plt.show()

print("Image statistics:")
widths = examples["width"]
heights = examples["height"]
print(f"Width  | avg (+/- std): {widths.mean():.2f} (+/- {widths.std():.2f}), min: {widths.min()}, max: {widths.max()}")
print(
    f"Height | avg (+/- std): {heights.mean():.2f} (+/- {heights.std():.2f}), min: {heights.min()}, max: {heights.max()}"
)


# %%
# Tune scan mask segmentation
def generate_scan_mask(image: np.ndarray):
    # Dynamic/adaptive thresholding to separate foreground
    threshold = skimage.filters.threshold_local(image, block_size=3)
    mask = image > threshold
    mask = mask > 0

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=500)

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=32)

    # Reflect the larger half of the mask to compensate for shadows
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_sum = np.sum(left_half)
    right_sum = np.sum(right_half)

    # Determine which half has the smaller sum and reflect it
    if left_sum > right_sum:
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)

    # Merge the two halves back together (pad/crop if image width is odd)
    mask = np.hstack((left_half, right_half))

    if mask.shape[1] < image.shape[1]:
        # Pad the mask by adding a single column of zeros to the right
        mask = np.hstack((mask, np.zeros((mask.shape[0], 1))))
    elif mask.shape[1] > image.shape[1]:
        # Crop the mask by removing the rightmost column
        mask = mask[:, :-1]

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


# Visualize the scan masks for the first 100 images
cols = 10
rows = 10

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    row = examples.iloc[i]
    image = io.imread(os.path.join(ROOT_DIR, row["image"]))

    if image.ndim == 3:
        image = image.mean(axis=-1).astype(np.uint8)

    mask = generate_scan_mask(image)
    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"

    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4, vmin=0, vmax=1)
    # ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
