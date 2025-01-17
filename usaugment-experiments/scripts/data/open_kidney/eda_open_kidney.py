"""The Open Kidney Ultrasound Dataset was published alongside the paper "The Open Kidney Ultrasound Data Set" (Singla
et al., 2023). The dataset contains 2D ultrasound images of the kidney, along with annotations for the kidney regions
and capsules. The annotations were created by two sonographers, who independently reviewed the images and provided
segmentation masks for the kidney regions and capsules. The images are also labeled acording to the quality of the
ultrasound image, the view, and whether the patient has received a kidney transplant.

The annotations and labels are not 100% consistent between the two sonographers. Since this dataset is one of many used
in our experiments, we will not attempt to resolve these inconsistencies. Instead, we will use the annotations and
labels from Sonographer 1.

Dataset: https://rsingla.ca/kidneyUS/
Paper: https://link.springer.com/chapter/10.1007/978-3-031-44521-7_15
"""

# %%
import ast
import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
from PIL import Image

ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/kidneyUS"

# %%
# Identify the duplicate images
image_filenames = [x for x in os.listdir(os.path.join(ROOT_DIR, "images")) if x.endswith(".png")]

filename_to_hash = {}
hash_to_filenames = {}

for filename in image_filenames:
    image = io.imread(os.path.join(ROOT_DIR, "images", filename))
    image_hash = hashlib.md5(image).hexdigest()
    filename_to_hash[filename] = image_hash

    if image_hash not in hash_to_filenames:
        hash_to_filenames[image_hash] = [filename]
    else:
        hash_to_filenames[image_hash].append(filename)

print(f"Number of unique images: {len(hash_to_filenames)}")

print("Identical images:")
for image_hash, filenames in hash_to_filenames.items():
    if len(filenames) > 1:
        print(f"\t{image_hash}: {filenames}")

duplicates = [f for h in hash_to_filenames.values() for f in h[1:] if len(h) > 1]

# %%
# Load the clinical data for Sonographer 1
clinical_data = pd.read_csv(os.path.join(ROOT_DIR, "labels/reviewed_labels_1.csv"))

# Parse the JSON strings in the "file_attributes" and "region_attributes" columns
clinical_data["quality"] = clinical_data["file_attributes"].apply(
    lambda x: ast.literal_eval(str(x)).get("Quality", None)
)
clinical_data["view"] = clinical_data["file_attributes"].apply(lambda x: ast.literal_eval(str(x)).get("View", None))
clinical_data["comments"] = clinical_data["file_attributes"].apply(
    lambda x: ast.literal_eval(str(x)).get("Comments", None)
)
clinical_data["anatomy"] = clinical_data["region_attributes"].apply(
    lambda x: ast.literal_eval(str(x)).get("Anatomy", None)
)
clinical_data.drop(columns=["file_attributes", "region_attributes"], inplace=True)

# Drop the duplicate images (keep the first occurrence)
clinical_data = clinical_data[~clinical_data["filename"].isin(duplicates)]

print(clinical_data.head().to_markdown())

# %%
# Plot a 6 x 4 grid of the "regions" annotated by Sonographer 1 and 2
for sonographer in [1, 2]:
    region_mask_dir = os.path.join(ROOT_DIR, f"labels/reviewed_masks_{sonographer}/regions")
    masks = os.listdir(region_mask_dir)

    fig, axes = plt.subplots(4, 6, figsize=(11, 6))
    for i, ax in enumerate(axes.flat):
        mask = Image.open(os.path.join(region_mask_dir, masks[i]))
        mask = np.array(mask)
        ax.imshow(mask, cmap="gray")
        ax.axis("off")

    plt.suptitle(f"Kidney Regions (Sonographer {sonographer})")
    plt.tight_layout()
    plt.show()

# %%
# Plot a 6 x 4 grid of the "capsules" annotated by Sonographer 1 and 2
for sonographer in [1, 2]:
    capsule_mask_dir = os.path.join(ROOT_DIR, f"labels/reviewed_masks_{sonographer}/capsule")
    masks = os.listdir(capsule_mask_dir)

    fig, axes = plt.subplots(4, 6, figsize=(11, 6))
    for i, ax in enumerate(axes.flat):
        mask = Image.open(os.path.join(capsule_mask_dir, masks[i]))
        mask = np.array(mask)
        ax.imshow(mask, cmap="gray")
        ax.axis("off")

    plt.suptitle(f"Kidney Capsules (Sonographer {sonographer})")
    plt.tight_layout()
    plt.show()

# %%
# Check that no entries are missing anatomy masks and that entries with missing anatomy labels have blank masks
image_filenames = [x.split(".")[0] for x in os.listdir(os.path.join(ROOT_DIR, "images")) if x.endswith(".png")]

missing_region_mask = []
missing_capsule_mask = []

region_mask_dir = os.path.join(ROOT_DIR, "labels/reviewed_masks_1/regions")
region_mask_filenames = [x.split(".")[0] for x in os.listdir(region_mask_dir) if x.endswith(".png")]

capsule_mask_dir = os.path.join(ROOT_DIR, "labels/reviewed_masks_1/capsule")
capsule_mask_filenames = [x.split(".")[0] for x in os.listdir(capsule_mask_dir) if x.endswith(".png")]

for image_filename in image_filenames:
    if image_filename not in region_mask_filenames:
        missing_region_mask.append(image_filename)
    if image_filename not in capsule_mask_filenames:
        missing_capsule_mask.append(image_filename)

print(f"Missing region masks: {len(missing_region_mask)}")
print(f"Missing capsule masks: {len(missing_capsule_mask)}")

# Verify masks
for i, row in clinical_data.iterrows():
    if row["anatomy"] == "Capsule":
        mask = io.imread(os.path.join(ROOT_DIR, "labels/reviewed_masks_1/capsule", row["filename"]))
    else:
        mask = io.imread(os.path.join(ROOT_DIR, "labels/reviewed_masks_1/regions", row["filename"]))

    if not row["anatomy"]:
        assert np.all(mask == 0), f"Mask for {row['filename']} is not blank"
    else:
        assert np.any(mask != 0), f"Mask for {row['filename']} is blank"

# %%
# Create canonical region IDs for each anatomy
anatomy_to_region_id = {
    "Capsule": 0,
    "Central Echo Complex": 1,
    "Medulla": 2,
    "Cortex": 3,
    "None": 4,
}

clinical_data["anatomy"] = clinical_data["anatomy"].fillna("None")
clinical_data["canonical_region_id"] = clinical_data["anatomy"].map(anatomy_to_region_id)

# %%
# Find entries with the same filename and canonical region ID
duplicate_entries = clinical_data[clinical_data.duplicated(subset=["filename", "canonical_region_id"], keep=False)]
print(duplicate_entries.to_markdown())

# %%
# Calculate attribute statistics
clinical_data["transplant"] = clinical_data["comments"].apply(lambda x: "transplant" in x)

subset_df = clinical_data.drop_duplicates("filename")

print("Total images:", len(subset_df))
print()
print(subset_df["transplant"].value_counts())
print()
print(subset_df["quality"].value_counts())
print()
print(subset_df["view"].value_counts())

# %%
# Look at the comments that mention "transplant"
uncertain_cases = []
for i, row in subset_df[subset_df["transplant"]].iterrows():
    print(row["comments"])

    if any(x in row["comments"].lower() for x in ["?", "probably", "might", "possibly", "possible"]):
        uncertain_cases.append(row["filename"])

print(f"Uncertain cases: {len(uncertain_cases)}")
print(uncertain_cases)
# %%
# Calculate image summary statistics
pixel_range = [np.inf, -np.inf]
heights = []
widths = []
aspect_ratios = []
channels = set()
labels = []

for filename in image_filenames:
    image = io.imread(os.path.join(ROOT_DIR, "images", f"{filename}.png"))

    min_pixel, max_pixel = np.min(image), np.max(image)
    pixel_range[0] = min(pixel_range[0], min_pixel)
    pixel_range[1] = max(pixel_range[1], max_pixel)
    heights.append(image.shape[0])
    widths.append(image.shape[1])
    aspect_ratios.append(image.shape[1] / image.shape[0])
    channels.add(image.shape[2] if len(image.shape) == 3 else 1)

# Plot the size of each image as a point on a scatter plot
fig, ax = plt.subplots()

ax.scatter(widths, heights, alpha=0.5)

ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
ax.set_aspect("equal")
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1200)
plt.title("Image Dimensions")
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
for filename in image_filenames:
    image = io.imread(os.path.join(ROOT_DIR, "images", f"{filename}.png"))

    if len(image.shape) == 2:
        continue
    elif not channels_are_identical(image):
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
    image = io.imread(os.path.join(ROOT_DIR, "images", f"{filename}.png"))

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
# Look at the unique mask values in the capsule and region masks

unique_capsule_values = set()
unique_region_values = set()

for filename in image_filenames:
    capsule_mask = io.imread(os.path.join(ROOT_DIR, "labels/reviewed_masks_1/capsule", f"{filename}.png"))
    region_mask = io.imread(os.path.join(ROOT_DIR, "labels/reviewed_masks_1/regions", f"{filename}.png"))

    unique_capsule_values.update(np.unique(capsule_mask))
    unique_region_values.update(np.unique(region_mask))

print(f"Unique capsule values: {unique_capsule_values}")
print(f"Unique region values: {unique_region_values}")


# %%
# Tune scan mask segmentation
def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Erode the mask
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Label the connected components
    label = skimage.measure.label(mask)

    # Keep the only the largest connected component
    regions = skimage.measure.regionprops(label)
    largest_region = max(regions, key=lambda x: x.area)
    mask = label == largest_region.label

    # Reflect the larger half of the mask to fill larger gaps in the fan and make it symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_sum = np.sum(left_half)
    right_sum = np.sum(right_half)

    # Determine which half has the greater sum and reflect it
    if left_sum > right_sum:
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


# Visualize the scan masks for the first 100 images
cols = 10
rows = 10

fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
indicies = np.arange(100)  # [2, 4, 7, 15, 18, 30]
for i, ax in enumerate(axes.flat):
    image = io.imread(os.path.join(ROOT_DIR, "images", f"{image_filenames[indicies[i]]}.png"))

    if image.ndim == 3:
        image = image.mean(axis=-1).astype(np.uint8)

    mask = generate_scan_mask(image)
    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"

    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4, vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
