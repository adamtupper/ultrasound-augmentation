"""The Annotated Ultrasound Liver dataset was published alongside the paper "Improving artificial intelligence pipeline
for liver malignancy diagnosis using ultrasound images and video frames" (Xu et al., 2023).

Dataset: https://zenodo.org/records/7272660
Paper: https://doi.org/10.1093/bib/bbac569
"""

# %%
# Setup
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io as io
from matplotlib.patches import Polygon

ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/Annotated_Ultrasound_Liver_Yiming/"
CLASS_DIRS = ["Normal", "Benign", "Malignant"]

# %%
# Plot the class distribution
normal_images = glob.glob(f"{ROOT_DIR}/Normal/image/*.jpg")
benign_images = glob.glob(f"{ROOT_DIR}/Benign/image/*.jpg")
malignant_images = glob.glob(f"{ROOT_DIR}/Malignant/image/*.jpg")

counts = {
    "Normal": len(normal_images),
    "Benign": len(benign_images),
    "Malignant": len(malignant_images),
}
total = sum(counts.values())

plt.bar(CLASS_DIRS, counts.values())
for i, count in enumerate(counts.values()):
    plt.text(i, count + 10, count, ha="center")
plt.text(0, 475, f"Total = {total}", ha="right")

plt.ylabel("Number of images")
plt.title("Class distribution")
plt.ylim(0, 500)
plt.show()

# %%
# Visualize a few images from each class
fig, axes = plt.subplots(3, 3, figsize=(12, 9))

for i, class_dir in enumerate(CLASS_DIRS):
    images = glob.glob(f"{ROOT_DIR}/{class_dir}/image/*.jpg")
    for j in range(3):
        image = io.imread(images[j])
        axes[i, j].imshow(image)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if j == 0:
            axes[i, j].set_ylabel(class_dir)

plt.tight_layout()
plt.show()

# %%
# Check the size of the images
image_dimensions = {x: [] for x in CLASS_DIRS}
for class_name in CLASS_DIRS:
    images = glob.glob(f"{ROOT_DIR}/{class_name}/image/*.jpg")
    for image_path in images:
        image = io.imread(image_path)
        image_dimensions[class_name].append(image.shape)

print("All 'normal' images have 3 channels?", np.all([x[2] == 3 for x in image_dimensions["Normal"]]))
print("All 'benign' images have 3 channels?", np.all([x[2] == 3 for x in image_dimensions["Benign"]]))
print("All 'malignant' images have 3 channels?", np.all([x[2] == 3 for x in image_dimensions["Malignant"]]))


# %%
# Check to see if the channels are all identical
def channels_are_identical(image):
    return np.all(image[:, :, :1] == image)


images_with_different_channels = []
for class_name in CLASS_DIRS:
    images = glob.glob(f"{ROOT_DIR}/{class_name}/image/*.jpg")
    for image_path in images:
        image = io.imread(image_path)

        if not channels_are_identical(image):
            images_with_different_channels.append(image_path)

print(f"Images with different channels ({len(images_with_different_channels)}):")
for image_path in images_with_different_channels[:10]:
    print(f"\t{image_path}")
print("\t...")


# %%
# Visualize the separate channels of the first 10 images with different channels and a
# heatmap of the pixel values that differ between the channels
def get_channel_differences(image):
    return (
        np.abs(image[:, :, 0] - image[:, :, 1]),
        np.abs(image[:, :, 0] - image[:, :, 2]),
        np.abs(image[:, :, 1] - image[:, :, 2]),
    )


fig, axes = plt.subplots(10, 4, figsize=(6, 12))

for i, image_path in enumerate(images_with_different_channels[:10]):
    image = io.imread(image_path)

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
# Plot the size of each image as a point on a scatterplot, coloured by class
fig, ax = plt.subplots()

symbols = ["o", "s", "^"]
for i, class_name in enumerate(CLASS_DIRS):
    x, y = zip(*[(x[1], x[0]) for x in image_dimensions[class_name]])
    ax.scatter(x, y, color=f"C{i}", marker=symbols[i], label=class_name, alpha=0.5)

ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
ax.set_aspect("equal")
ax.set_xlim(0, 1500)
ax.set_ylim(0, 1500)
ax.legend()

plt.title("Image dimensions by class")
plt.show()

print("Image statistics:")
print(f"Width  | avg (+/- std): {np.mean(x):.2f} (+/- {np.std(x):.2f}), min: {np.min(x)}, max: {np.max(x)}")
print(f"Height | avg (+/- std): {np.mean(y):.2f} (+/- {np.std(y):.2f}), min: {np.min(y)}, max: {np.max(y)}")


# %%
# Load images, class labels, and segmentation masks
def load_mask_coords(mask_path):
    with open(mask_path) as f:
        mask = json.load(f)

    return mask


examples = []
for class_name in CLASS_DIRS:
    images = glob.glob(f"{ROOT_DIR}/{class_name}/image/*.jpg")
    for image_path in images:
        filename = os.path.basename(image_path).removesuffix(".jpg")
        scan_outline_path = os.path.join(ROOT_DIR, class_name, "segmentation/outline", f"{filename}.json")
        liver_outline_path = os.path.join(ROOT_DIR, class_name, "segmentation/liver", f"{filename}.json")
        mass_outline_path = os.path.join(ROOT_DIR, class_name, "segmentation/mass", f"{filename}.json")

        examples.append(
            {
                "filename": os.path.basename(image_path),
                "class": class_name,
                "image": io.imread(image_path),
                "scan_outline_coords": (
                    load_mask_coords(scan_outline_path) if os.path.exists(scan_outline_path) else None
                ),
                "liver_coords": load_mask_coords(liver_outline_path) if os.path.exists(liver_outline_path) else None,
                "mass_coords": load_mask_coords(mass_outline_path) if os.path.exists(mass_outline_path) else None,
            }
        )

print("Normal examples with...")
missing_scan = [x["filename"] for x in examples if (x["class"] == "Normal") and (x["scan_outline_coords"] is None)]
missing_liver = [x["filename"] for x in examples if (x["class"] == "Normal") and (x["liver_coords"] is None)]
print(f"\tmissing scan outline: {missing_scan}")
print(f"\tmissing liver outline: {missing_liver}")

print("Benign examples with...")
missing_scan = [x["filename"] for x in examples if (x["class"] == "Benign") and (x["scan_outline_coords"] is None)]
missing_liver = [x["filename"] for x in examples if (x["class"] == "Benign") and (x["liver_coords"] is None)]
missing_mass = [x["filename"] for x in examples if (x["class"] == "Benign") and (x["mass_coords"] is None)]
print(f"\tmissing scan outline: {missing_scan}")
print(f"\tmissing liver outline: {missing_liver}")
print(f"\tmissing mass outline: {missing_mass}")

print("Malignant examples with...")
missing_scan = [x["filename"] for x in examples if (x["class"] == "Malignant") and (x["scan_outline_coords"] is None)]
missing_liver = [x["filename"] for x in examples if (x["class"] == "Malignant") and (x["liver_coords"] is None)]
missing_mass = [x["filename"] for x in examples if (x["class"] == "Malignant") and (x["mass_coords"] is None)]
print(f"\tmissing scan outline: {missing_scan}")
print(f"\tmissing liver outline: {missing_liver}")
print(f"\tmissing mass outline: {missing_mass}")

# %%
# Visualize the scan, liver, and mass region masks for a few images from each class
normal_idxs = [i for i, x in enumerate(examples) if x["class"] == "Normal"]
benign_idxs = [i for i, x in enumerate(examples) if x["class"] == "Benign"]
malignant_idxs = [i for i, x in enumerate(examples) if x["class"] == "Malignant"]

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for i, (class_name, idxs) in enumerate(zip(CLASS_DIRS, [normal_idxs, benign_idxs, malignant_idxs])):
    for j in range(3):
        ax = axes[i, j]
        example = examples[idxs[j]]

        ax.imshow(example["image"])
        if example["scan_outline_coords"] is not None:
            ax.add_patch(
                Polygon(
                    example["scan_outline_coords"],
                    edgecolor="grey",
                    facecolor="none",
                    linewidth=1.5,
                    label="Scan",
                )
            )
        if example["liver_coords"] is not None:
            ax.add_patch(
                Polygon(
                    example["liver_coords"],
                    edgecolor="red",
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="dashed",
                    label="Liver",
                )
            )
        if example["mass_coords"] is not None:
            ax.add_patch(
                Polygon(
                    example["mass_coords"],
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="dashdot",
                    label="Mass",
                )
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        if j == 0:
            ax.set_ylabel(class_name)


plt.tight_layout()
plt.show()

# %%
# Look at the annotations of the malignant examples with missing liver outlines
malignant_missing_liver_idxs = [i for i in malignant_idxs if examples[i]["liver_coords"] is None]

fig, axes = plt.subplots(1, len(malignant_missing_liver_idxs), figsize=(12, 4))

for i, idx in enumerate(malignant_missing_liver_idxs):
    ax = axes[i]
    example = examples[idx]

    ax.imshow(example["image"])
    if example["scan_outline_coords"] is not None:
        ax.add_patch(
            Polygon(
                example["scan_outline_coords"],
                edgecolor="grey",
                facecolor="none",
                linewidth=1.5,
                label="Scan",
            )
        )
    if example["mass_coords"] is not None:
        ax.add_patch(
            Polygon(
                example["mass_coords"],
                edgecolor="yellow",
                facecolor="none",
                linewidth=1.5,
                linestyle="dashdot",
                label="Mass",
            )
        )

    ax.set_title(f"{example['filename']} ({example['class']})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

print(
    """
    Images 229.jpg and 306.jpg are actually missing the scan outlines, not the liver outline. The liver outline has
    been erroneously saved as the scan outline.
    """
)

# %%
# Test mask generation on a few examples from each class
normal_idxs = [i for i, x in enumerate(examples) if x["class"] == "Normal"]
benign_idxs = [i for i, x in enumerate(examples) if x["class"] == "Benign"]
malignant_idxs = [i for i, x in enumerate(examples) if x["class"] == "Malignant"]

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for i, (class_name, idxs) in enumerate(zip(CLASS_DIRS, [normal_idxs, benign_idxs, malignant_idxs])):
    for j in range(3):
        ax = axes[i, j]
        example = examples[idxs[j]]
        height, width = example["image"].shape[:2]

        ax.imshow(example["image"])
        if example["scan_outline_coords"] is not None:
            coords = [[y, x] for x, y in example["scan_outline_coords"]]
            mask = skimage.draw.polygon2mask((height, width), coords)
            ax.imshow(mask, alpha=0.5)
            ax.add_patch(
                Polygon(
                    example["scan_outline_coords"],
                    edgecolor="grey",
                    facecolor="none",
                    linewidth=1.5,
                    label="Scan",
                )
            )
        if example["liver_coords"] is not None:
            coords = [[y, x] for x, y in example["liver_coords"]]
            mask = skimage.draw.polygon2mask((height, width), coords)
            ax.imshow(mask, alpha=0.5)
            ax.add_patch(
                Polygon(
                    example["liver_coords"],
                    edgecolor="red",
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="dashed",
                    label="Liver",
                )
            )
        if example["mass_coords"] is not None:
            coords = [[y, x] for x, y in example["mass_coords"]]
            mask = skimage.draw.polygon2mask((height, width), coords)
            ax.imshow(mask, alpha=0.5)
            ax.add_patch(
                Polygon(
                    example["mass_coords"],
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="dashdot",
                    label="Mass",
                )
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        if j == 0:
            ax.set_ylabel(class_name)


plt.tight_layout()
plt.show()


# %%
# Generate scan masks for images 229.jpg and 306.jpg
def segment_fan(image: np.ndarray):
    # Threshold the image
    mask = image > 1

    # Morphological operations
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    mask = skimage.morphology.remove_small_objects(mask, min_size=1000)

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask


# Load the images
image_229 = io.imread(os.path.join(ROOT_DIR, "Malignant/image/229.jpg")).mean(axis=-1)
image_306 = io.imread(os.path.join(ROOT_DIR, "Malignant/image/306.jpg")).mean(axis=-1)

# Generate the segmentation masks
mask_229 = segment_fan(image_229)
mask_306 = segment_fan(image_306)

# Visualize the masks
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_229, cmap="gray")
axes[0].imshow(mask_229, alpha=0.4)
axes[0].set_title("229.jpg")
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].imshow(image_306, cmap="gray")
axes[1].imshow(mask_306, alpha=0.4)
axes[1].set_title("306.jpg")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()

# %%
