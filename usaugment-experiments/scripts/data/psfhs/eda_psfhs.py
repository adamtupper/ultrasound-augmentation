"""The PSFHS dataset was published in the paper "PSFHS: Intrapartum ultrasound  image dataset for aI-based  segmentation
of pubic symphysis  and fetal head" (Chen et al., 2024).

Dataset: https://zenodo.org/records/10969427
Paper: https://www.nature.com/articles/s41597-024-03266-4
"""

# %%
# Setup
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage

ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/PSFHS"

# %%
# Find out how many channels are in the images and masks
image_shapes = []
for image_path in glob.glob(os.path.join(ROOT_DIR, "image_mha", "*.mha")):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    image_shapes.append(image.shape)

print("# images with N channels:")
channels, counts = np.unique([shape[0] for shape in image_shapes], return_counts=True)
for n, count in zip(channels, counts):
    print(f"{n}: {count}")

mask_shapes = []
for mask_path in glob.glob(os.path.join(ROOT_DIR, "label_mha", "*.mha")):
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    mask_shapes.append(mask.shape if mask.ndim == 3 else (1,) + mask.shape)

print("# masks with N channels:")
channels, counts = np.unique([shape[0] for shape in mask_shapes], return_counts=True)
for n, count in zip(channels, counts):
    print(f"{n}: {count}")

# %%
# Visualize the first 10 images and their segmentation masks

image_paths = glob.glob(os.path.join(ROOT_DIR, "image_mha", "*.mha"))
mask_paths = glob.glob(os.path.join(ROOT_DIR, "label_mha", "*.mha"))

fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i, (image_path, mask_path) in enumerate(zip(image_paths[:10], mask_paths[:10])):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).transpose(1, 2, 0)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

    axes[0, i].imshow(image, cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(mask, cmap="gray")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()

# %%
# Are the pixel values in each channel of each image identical?
image_paths = glob.glob(os.path.join(ROOT_DIR, "image_mha", "*.mha"))
for image_path in image_paths:
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).transpose(1, 2, 0)

    channels_are_identical = np.all(image[:, :, :1] == image)
    if not channels_are_identical:
        raise ValueError(f"The channels are not identical in at least one image ({os.path.basename(image_path)})!")

print("All channels are identical in all images!")

# %%
# Create a dataframe of the image and mask statistics
examples = []
for image_path in glob.glob(os.path.join(ROOT_DIR, "image_mha", "*.mha")):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    channels, height, width = image.shape

    mask_path = image_path.replace("image_mha", "label_mha")
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    contains_ps = 1 in np.unique(mask)
    contains_fh = 2 in np.unique(mask)

    examples.append(
        {
            "image_path": image_path.removeprefix(ROOT_DIR + "/"),
            "mask_path": mask_path.removeprefix(ROOT_DIR + "/"),
            "image_channels": channels,
            "image_height": height,
            "image_width": width,
            "contains_ps": contains_ps,
            "contains_fh": contains_fh,
        }
    )

metadata = pd.DataFrame(examples)
metadata.head()

# %%
# Count the number of images, and the number of images with PS and FH
print(f"Number of images: {metadata.shape[0]}")
print(f"Number of images with PS: {metadata['contains_ps'].sum()}")
print(f"Number of images with FH: {metadata['contains_fh'].sum()}")

# %%
# Print the image summary statistics (height, width, channels, etc.)
print("Image statistics:")
print(metadata.describe())


# %%
# Tune scan mask segmentation
def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 1

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=64)

    # # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Erode the mask
    for i in range(3):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for i in range(3):
        mask = skimage.morphology.binary_dilation(mask)

    # Take the union of the right half reflected onto the left half, and the left half reflected onto the right half
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    # left_sum = np.sum(left_half)
    # right_sum = np.sum(right_half)

    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    # # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


# Visualize the scan masks for the first 100 images
cols = 10
rows = 10

files = glob.glob(os.path.join(ROOT_DIR, "image_mha", "*.mha"))

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    image = sitk.GetArrayFromImage(sitk.ReadImage(files[i + 200]))

    if image.ndim == 3:
        image = image.mean(axis=0).astype(np.uint8)

    mask = generate_scan_mask(image)
    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"

    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4, vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
