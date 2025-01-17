"""Stanford Thyroid Ultrasound Cine-clip Dataset

Dataset: https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5
"""

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage

DATA_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/thyroidultrasoundcineclip"

# %%
# General statistics
f = h5py.File(os.path.join(DATA_DIR, "dataset.hdf5"), "r")

for key in list(f.keys()):
    print(key, f[key].shape, f[key].dtype)

print(f"Min/Max image pixel values: {np.min(f['image'][0])}, {np.max(f['image'][0])}")
print(f"Unique mask pixel values: {np.unique(f['mask'][:10])}")
print(f"Image shape: {f['image'][0].shape}, Mask shape: {f['mask'][0].shape}")

# %%
# Plot the first 10 images and masks
fig, axes = plt.subplots(2, 10, figsize=(10, 2.1))

for i in range(10):
    axes[0, i].imshow(f["image"][i], cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Image")

    axes[1, i].imshow(f["mask"][i], cmap="gray")
    axes[1, i].axis("off")
    axes[1, i].set_title("Mask")

plt.tight_layout()
plt.show()

# %%
# Inspect the metadata
metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
metadata.head()

# %%
# Plot the age, sex, and ti-rads_level distributions for each set
for variable in ["age", "sex", "ti-rads_level"]:
    grid = sns.catplot(data=metadata, x=variable, kind="count", height=4, aspect=2)
    grid.set_axis_labels(variable, "# Examples")
    grid.set_titles("{col_name}")
    (grid.figure.suptitle(f"{variable.capitalize()} Distribution"),)

    if variable == "age":
        grid.set_xticklabels(step=5)

    for ax in grid.axes.ravel():
        # add annotations
        for c in ax.containers:
            ax.bar_label(c, label_type="edge")
        ax.margins(y=0.2)

    plt.show()

# %%
# Count the number of frames for each patient
annot_ids, counts = np.unique(f["annot_id"], return_counts=True)
plt.hist(counts)
plt.xlabel("# Frames")
plt.ylabel("# Patients")
plt.title("Distribution of the Number of Frames per Patient")
plt.show()


# %%
# Count the number of patients
print(f"Number of patients: {len(annot_ids)}")

# Find the first image index for each patient
unique_annot_ids = np.unique(f["annot_id"])
patient_indices = np.array([np.where(f["annot_id"][:] == annot_id)[0][0] for annot_id in unique_annot_ids])

# Plot the first image for each patient in a 10 x 20 grid
cols = 10
rows = 20

fig, axes = plt.subplots(rows, cols, figsize=(24, 48))
for i, ax in enumerate(axes.flat):
    if i >= len(patient_indices):
        ax.axis("off")
        continue

    image = f["image"][patient_indices[i]]

    if image.ndim == 3:
        image = image.mean(axis=0).astype(np.uint8)

    ax.imshow(image, cmap="gray")
    ax.set_title(unique_annot_ids[i].decode("utf-8"))
    ax.axis("off")

plt.tight_layout()
plt.show()


# %%
# Tune scan mask segmentation
def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 1

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=64)

    # Erode the mask
    for _ in range(30):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for _ in range(30):
        mask = skimage.morphology.binary_dilation(mask)

    # Logical OR of left and right halves to mitigate shadows
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]

    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


# Load the IDs of the patients whose scans were acquired using a convex probe
patients = pd.read_csv(os.path.join("patients.csv"))

# Find the index of the first frame for each patient
annot_ids = [x.decode("utf-8") for x in f["annot_id"]]
indices = []
for patient_id in patients["patient"]:
    for i, annot_id in enumerate(annot_ids):
        if annot_id == f"{patient_id}_":
            indices.append(i)
            break

# Visualize the scan masks for each frame
cols = 10
rows = 11
fig, axes = plt.subplots(rows, cols, figsize=(24, 36))
for i, ax in enumerate(axes.flat):
    if i > len(indices) - 1:
        ax.axis("off")
        continue

    image = f["image"][indices[i]]

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
