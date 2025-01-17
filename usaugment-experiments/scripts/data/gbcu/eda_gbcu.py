"""The Gallbladder Cancer Ultrasound dataset was published alongside the paper "Surpassing the Human Accuracy: Detecting
Gallbladder Cancer from USG Images with Curriculum Learning" (Basu et al., 2022).

Dataset: https://gbc-iitd.github.io/data/gbcu
Paper: https://ieeexplore.ieee.org/document/9879895
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import skimage.io as io

DATA_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/GBCU-Shared"

# %%
# Plot class distribution
train_examples = pd.read_csv(f"{DATA_DIR}/train.txt", header=None, names=["filename", "label"])
train_examples["split"] = "train"
test_examples = pd.read_csv(f"{DATA_DIR}/test.txt", header=None, names=["filename", "label"])
test_examples["split"] = "test"
examples = pd.concat([train_examples, test_examples])

g = sns.catplot(data=examples, x="label", kind="count", hue="split")
plt.title("Class Distribution")
plt.xticks([0, 1, 2], ["Normal", "Benign", "Malignant"])
g.set_axis_labels("Class", "# Examples")
plt.show()

# %%
# Plot the first 10 images
filenames = os.listdir(f"{DATA_DIR}/imgs")
filenames = sorted([image for image in filenames if not image.startswith(".")])
filenames = filenames[:10]

fig, axes = plt.subplots(1, len(filenames), figsize=(15, 3))
for filename, ax in zip(filenames, axes):
    image = io.imread(f"{DATA_DIR}/imgs/{filename}")
    ax.imshow(image, cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()
# %%
# Print summary statistics for the images (e.g., heights, widths, aspect ratios, min/max pixel values)
pixel_range = [np.inf, -np.inf]
heights = []
widths = []
aspect_ratios = []
channels = set()
labels = []

filenames = [x for x in os.listdir(f"{DATA_DIR}/imgs") if not x.startswith(".")]
for filename in filenames:
    image = io.imread(f"{DATA_DIR}/imgs/{filename}")

    labels.append(examples[examples["filename"] == filename]["label"].values[0])
    min_pixel, max_pixel = np.min(image), np.max(image)
    pixel_range[0] = min(pixel_range[0], min_pixel)
    pixel_range[1] = max(pixel_range[1], max_pixel)
    heights.append(image.shape[0])
    widths.append(image.shape[1])
    aspect_ratios.append(image.shape[1] / image.shape[0])
    channels.add(image.shape[2] if len(image.shape) == 3 else 1)

# Plot the size of each image as a point on a scatter plot, colourer by class
fig, ax = plt.subplots()

for label, marker in zip([0, 1, 2], ["o", "s", "^"]):
    indices = [i for i, x in enumerate(labels) if x == label]
    ax.scatter([widths[i] for i in indices], [heights[i] for i in indices], label=label, alpha=0.5, marker=marker)
ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
ax.set_aspect("equal")
ax.set_xlim(0, 1700)
ax.set_ylim(0, 1700)
ax.legend()
plt.title("Image Dimensions by Class")
plt.show()

print(f"Pixel range: {pixel_range}")
print(f"Height: mean={np.mean(heights):.2f}, std={np.std(heights):.2f}, min={np.min(heights)}, max={np.max(heights)}")
print(f"Width: mean={np.mean(widths):.2f}, std={np.std(widths):.2f}, min={np.min(widths)}, max={np.max(widths)}")
print(f"Aspect ratio: mean={np.mean(aspect_ratios):.2f}, std={np.std(aspect_ratios):.2f}")
print(f"Channels: {channels}")


# %%
# Tune scan mask segmentation
def segment_fan(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)

    # Erode the mask
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


rows = 10
cols = 10

# Visualize the scan mask of the first 100 images
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    filename = filenames[i]
    image = io.imread(f"{DATA_DIR}/imgs/{filename}")
    if image.ndim == 3:
        image = image.mean(axis=-1)
    mask = segment_fan(image)
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4, vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
