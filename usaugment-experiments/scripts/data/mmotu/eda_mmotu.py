"""The Multi-Modality Ovarian Tumor Ultrasound Image (MMOTU) Dataset was published alongside the paper "MMOTU: A
Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation" (Zhao et al.,
2023).

Dataset: https://github.com/cv516Buaa/MMOTU_DS2Net
Paper: http://arxiv.org/abs/2207.06799
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.io as io

DATA_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/MMOTU"

# %%
# How many different classes are there?
label_df = pd.read_csv(
    os.path.join(DATA_DIR, "OTU_2d", "train_cls.txt"), delimiter="  ", names=["image", "label"], engine="python"
)
unique_labels = label_df["label"].unique()
print(f"Unique labels: {unique_labels}")

# %% Calculate summary statistics for the images (e.g., heights, widths, aspect ratios, min/max pixel values)
train_examples = pd.read_csv(
    os.path.join(DATA_DIR, "OTU_2d", "train_cls.txt"), delimiter="  ", names=["image", "label"], engine="python"
)
test_examples = pd.read_csv(
    os.path.join(DATA_DIR, "OTU_2d", "val_cls.txt"), delimiter="  ", names=["image", "label"], engine="python"
)
examples = pd.concat([train_examples, test_examples])

pixel_range = [np.inf, -np.inf]
heights = []
widths = []
aspect_ratios = []
channels = set()
labels = []

for i, row in examples.iterrows():
    filename, label = row["image"], row["label"]
    image = io.imread(os.path.join(DATA_DIR, "OTU_2d", "images", filename))

    labels.append(label)
    min_pixel, max_pixel = np.min(image), np.max(image)
    pixel_range[0] = min(pixel_range[0], min_pixel)
    pixel_range[1] = max(pixel_range[1], max_pixel)
    heights.append(image.shape[0])
    widths.append(image.shape[1])
    aspect_ratios.append(image.shape[1] / image.shape[0])
    channels.add(image.shape[2] if len(image.shape) == 3 else 1)

# Plot the size of each image as a point on a scatter plot, coloured by class
fig, ax = plt.subplots()

for i, (label, marker) in enumerate(zip(np.unique(labels), [".", "o", "v", "^", "<", ">", "s", "*"])):
    indices = np.where(labels == label)
    ax.scatter(
        np.array(widths)[indices],
        np.array(heights)[indices],
        label=label,
        marker=marker,
        alpha=0.5,
    )

ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
ax.set_aspect("equal")
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1200)
ax.legend()
plt.title("Image Dimensions by Class")
plt.show()

# Print summary statistics
print(f"Pixel range: {pixel_range}")
print(f"Height: mean={np.mean(heights):.2f}, std={np.std(heights):.2f}, min={np.min(heights)}, max={np.max(heights)}")
print(f"Width: mean={np.mean(widths):.2f}, std={np.std(widths):.2f}, min={np.min(widths)}, max={np.max(widths)}")
print(f"Aspect ratio: mean={np.mean(aspect_ratios):.2f}, std={np.std(aspect_ratios):.2f}")
print(f"Channels: {channels}")


# %%
# Check to see if the channels are all identical
def channels_are_identical(image):
    return np.all(image[:, :, :1] == image)


images_with_different_channels = []
for i, row in examples.iterrows():
    filename = row["image"]
    image = io.imread(os.path.join(DATA_DIR, "OTU_2d", "images", filename))

    if not channels_are_identical(image):
        images_with_different_channels.append(filename)

print(f"Images with different channels ({len(images_with_different_channels)}):")
for image_path in images_with_different_channels[:10]:
    print(f"\t{image_path}")
print("\t...")


# %%
# Visualize the separate channels of the first 10 images with different channels and a heatmap of the pixel values that
# differ between the channels
def get_channel_differences(image):
    return (
        np.abs(image[:, :, 0] - image[:, :, 1]),
        np.abs(image[:, :, 0] - image[:, :, 2]),
        np.abs(image[:, :, 1] - image[:, :, 2]),
    )


fig, axes = plt.subplots(10, 4, figsize=(6, 12))

for i, filename in enumerate(images_with_different_channels[:10]):
    image = io.imread(os.path.join(DATA_DIR, "OTU_2d", "images", filename))

    for j in range(3):
        axes[i, j].imshow(image[:, :, j], cmap="gray")
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j].set_xlabel(f"Channel {j}")

    r_diff, g_diff, b_diff = get_channel_differences(image)
    axes[i, 3].imshow(np.stack([r_diff, g_diff, b_diff], axis=-1))
    axes[i, 3].set_xticks([])
    axes[i, 3].set_yticks([])
    axes[i, 3].set_xlabel("Difference")

plt.tight_layout()
plt.show()

# %%
# Plot the first 10 training images, their masks, and their labels

label_df = pd.read_csv(
    os.path.join(DATA_DIR, "OTU_2d", "train_cls.txt"), delimiter="  ", names=["image", "label"], engine="python"
)

num_images = 10
fig, axes = plt.subplots(4, 10, figsize=(15, 7))

for row, example_axes in zip(label_df.itertuples(), axes.transpose()):
    filename = row.image.removesuffix(".JPG")
    image_path = os.path.join(DATA_DIR, "OTU_2d", "images", f"{filename}.JPG")
    mask_path = os.path.join(DATA_DIR, "OTU_2d", "annotations", f"{filename}.PNG")
    binary_mask_path = os.path.join(DATA_DIR, "OTU_2d", "annotations", f"{filename}_binary.PNG")
    binary_binary_mask_path = os.path.join(DATA_DIR, "OTU_2d", "annotations", f"{filename}_binary_binary.PNG")
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    binary_mask = io.imread(binary_mask_path)
    binary_binary_mask = io.imread(binary_binary_mask_path)

    mask_values = np.unique(mask.reshape(-1, 3), axis=0)[1:]  # Exclude the background
    binary_mask_values = np.unique(binary_mask)[1:]  # Exclude the background
    binary_binary_mask_values = np.unique(binary_binary_mask)[1:]  # Exclude the background

    example_axes[0].imshow(image, cmap="gray")
    example_axes[1].imshow(mask, cmap="gray")
    example_axes[2].imshow(binary_mask, cmap="gray")
    example_axes[3].imshow(binary_binary_mask, cmap="gray")

    example_axes[0].axis("off")
    example_axes[1].axis("off")
    example_axes[2].axis("off")
    example_axes[3].axis("off")
    example_axes[0].set_title(f"Label: {row.label}")
    example_axes[1].set_title(mask_values)
    example_axes[2].set_title(binary_mask_values)
    example_axes[3].set_title(binary_binary_mask_values)

plt.tight_layout()
plt.show()

# %%
# Print the unique pixel values in the masks
unique_values = set()
for i, row in enumerate(label_df.itertuples()):
    mask_path = os.path.join(DATA_DIR, "OTU_2d", "annotations", f"{row.image.removesuffix('.JPG')}.PNG")
    mask = io.imread(mask_path)
    new_values = [tuple(x) for x in np.unique(mask.reshape(-1, 3), axis=0)]
    unique_values.update(new_values)

    if i == 9:
        break

print(f"Unique values in masks ({len(unique_values)}):")
for value in unique_values:
    print("\t", value)

unique_values = set()
for i, row in enumerate(label_df.itertuples()):
    binary_mask_path = os.path.join(DATA_DIR, "OTU_2d", "annotations", f"{row.image.removesuffix('.JPG')}_binary.PNG")
    binary_mask = io.imread(binary_mask_path)
    new_values = np.unique(binary_mask)
    unique_values.update(new_values)

    if i == 9:
        break

print(f"Unique values in binary masks ({len(unique_values)}):")
for value in unique_values:
    print("\t", value)


unique_values = set()
for i, row in enumerate(label_df.itertuples()):
    binary_binary_mask_path = os.path.join(
        DATA_DIR, "OTU_2d", "annotations", f"{row.image.removesuffix('.JPG')}_binary_binary.PNG"
    )
    binary_binary_mask = io.imread(binary_binary_mask_path)
    new_values = np.unique(binary_binary_mask)
    unique_values.update(new_values)

    if i == 9:
        break

print(f"Unique values in binary binary masks ({len(unique_values)}):")
for value in unique_values:
    print("\t", value)

# %%
# Print the unique pixel values for the masks in the training set for each class
label_df = pd.read_csv(
    os.path.join(DATA_DIR, "OTU_2d", "train_cls.txt"), delimiter="  ", names=["image", "label"], engine="python"
)

for label in label_df["label"].unique():
    unique_values = set()
    for i, row in enumerate(label_df.itertuples()):
        if row.label != label:
            continue

        mask_path = os.path.join(DATA_DIR, "OTU_2d", "annotations", f"{row.image.removesuffix('.JPG')}.PNG")
        mask = io.imread(mask_path)
        unique_values.update([tuple(x) for x in np.unique(mask.reshape(-1, 3), axis=0)])

    print(f"Unique values in masks for label {label} ({len(unique_values)}):")
    for value in unique_values:
        print("\t", value)

# %%
# Try mapping the pixel values of a multi-class mask to the class indices
PIXEL_TO_CLASS = {
    (0, 0, 0): 0,  # Background
    (64, 0, 0): 1,  # Chocolate cyst
    (0, 64, 0): 2,  # Serious cystadenoma
    (0, 0, 64): 3,  # Teratoma
    (64, 0, 64): 4,  # Theca cell tumor
    (64, 64, 0): 5,  # Simple cyst
    (64, 64, 64): 6,  # Normal ovary
    (0, 128, 0): 7,  # Mucinous cystadenoma
    (0, 0, 128): 8,  # High grade serous
}

transformed_mask = np.apply_along_axis(lambda x: PIXEL_TO_CLASS[tuple(x)], axis=-1, arr=mask)
print(transformed_mask.shape)
print(np.unique(transformed_mask))


# %%
# Tune scan segmentation
def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Erode the mask
    for i in range(10):
        mask = skimage.morphology.binary_erosion(mask)

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=5000)

    # Dilate the mask
    for i in range(10):
        mask = skimage.morphology.binary_dilation(mask)

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Reflect the smaller half of the mask to remove annoations fill and make the scan symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_sum = np.sum(left_half)
    right_sum = np.sum(right_half)

    # Determine which half has the smaller sum and reflect it
    if left_sum < right_sum:
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)

    # Merge the two halves back together
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


# Visualize the scan masks of the first 100 images
filenames = label_df["image"].values[:100]

cols = 10
rows = 10

fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    image = io.imread(os.path.join(DATA_DIR, "OTU_2d", "images", filenames[i]))
    if image.ndim == 3:
        image = image.mean(axis=-1)
    mask = generate_scan_mask(image)
    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4, vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
