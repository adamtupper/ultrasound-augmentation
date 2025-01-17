"""The Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS) dataset was published alongside the
paper "Deep Learning for Segmentation using an Open  Large-Scale Dataset in 2D Echocardiography" (Leclerc et al., 2019).

Dataset: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html
Paper: https://ieeexplore.ieee.org/document/8649738/
"""

# %%
import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import skimage
import yaml

DATA_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/CAMUS_public"


# %%
# Extract metadata from the *.cfg files
def extract_metadata(dataset_dir):
    """Extract the metadata from the ".cfg" files."""
    metadata = []
    for file_path in glob.glob(os.path.join(dataset_dir, "database_nifti", "**", "*.cfg")):
        patient_id, filename = file_path.split("/")[-2:]
        patient_id = int(patient_id.removeprefix("patient"))
        view = filename.removesuffix(".cfg").split("_")[-1]

        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        for i in range(config["NbFrame"]):
            frame = i + 1
            row = {
                "patient": patient_id,
                "view": view,
                "frame": frame,
                "sex": config["Sex"],
                "age": config["Age"],
                "image_quality": config["ImageQuality"],
                "EF": config["EF"],
                "frame_rate": config["FrameRate"],
            }
            row["ED"] = True if config["ED"] == frame else False
            row["ES"] = True if config["ES"] == frame else False
            metadata.append(row)

    df = pd.DataFrame(metadata)

    return df


metadata_df = extract_metadata(DATA_DIR)
metadata_df.head()

# %%
# Plot the distribution of image quality
counts = metadata_df["image_quality"].value_counts()
total = counts.sum()

plt.bar(counts.index, counts)
for i, count in enumerate(counts):
    plt.text(i, count + 100, f"{count} ({count/total:.1%})", ha="center", va="bottom")
plt.text(2, 11000, f"Total = {total}", ha="center")

plt.ylabel("Image count")
plt.xlabel("Image quality")
plt.ylim(0, 12000)
plt.title("Distribution of image quality")
plt.show()

# %%
# Plot the stills for a single patient
patient = "patient0001"

fig, axes = plt.subplots(2, 4)
for i, suffix in enumerate(["", "_gt"]):
    for j, view in enumerate(["2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"]):
        image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}{suffix}.nii.gz")
        axes[i, j].imshow(image_file.get_fdata(), cmap="gray")
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(view)

plt.tight_layout()
plt.show()

# %%
# Plot the 2-chamber and 4-chamber sequences for a single patient
patient = "patient0001"

key_frames = []
for view in ["2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"]:
    image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}.nii.gz")
    key_frames.append(image_file.get_fdata())

for sequence in ["2CH", "4CH"]:
    image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{sequence}_half_sequence.nii.gz")
    mask_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{sequence}_half_sequence_gt.nii.gz")
    images = image_file.get_fdata()
    masks = mask_file.get_fdata()

    fig, axes = plt.subplots(2, images.shape[2], figsize=(10.5, 2))
    for i in range(images.shape[2]):
        axes[0, i].imshow(images[:, :, i], cmap="gray")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].imshow(masks[:, :, i], cmap="gray")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        for frame in key_frames:
            if np.all(images[:, :, i] == frame):
                axes[0, i].spines[:].set_color("red")
                axes[0, i].spines[:].set_linewidth(2)
                axes[1, i].spines[:].set_color("red")
                axes[1, i].spines[:].set_linewidth(2)

    plt.suptitle(sequence)
    plt.tight_layout()
    plt.show()

# %%
# Plot the frames from three poor, medium, and good quality sequences
sequences_4ch_df = metadata_df[metadata_df["view"] == "4CH"].drop_duplicates(subset=["patient"], keep="first")

# Sample three sequences from each quality category
poor_4ch_sequences = sequences_4ch_df[sequences_4ch_df["image_quality"] == "Poor"]["patient"].sample(3).values
medium_4ch_sequences = sequences_4ch_df[sequences_4ch_df["image_quality"] == "Medium"]["patient"].sample(3).values
good_4ch_sequences = sequences_4ch_df[sequences_4ch_df["image_quality"] == "Good"]["patient"].sample(3).values
sequences = np.concatenate([poor_4ch_sequences, medium_4ch_sequences, good_4ch_sequences])

# Display the first 10 image in each sequence
fig, axes = plt.subplots(9, 10, figsize=(10, 10))
for i, patient in enumerate(sequences):
    row = axes[i]

    if patient in poor_4ch_sequences:
        row[0].set_ylabel(f"Poor ({patient})")
    elif patient in medium_4ch_sequences:
        row[0].set_ylabel(f"Medium ({patient})")
    else:
        row[0].set_ylabel(f"Good ({patient})")

    patient = f"patient{patient:04d}"
    image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_4CH_half_sequence.nii.gz")
    images = image_file.get_fdata()

    for j in range(10):
        ax = row[j]
        ax.imshow(images[:, :, i], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()

# %%
# Check the range of pixels values
patient = "patient0001"
view = "2CH_ED"

image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}.nii.gz")
print("Images:", image_file.get_fdata().dtype)

mask_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}_gt.nii.gz")
print("Masks:", mask_file.get_fdata().dtype)
print()
print("Unique mask values:", np.unique(mask_file.get_fdata()))
print("Unique image values:", np.unique(image_file.get_fdata()))


# %%
# Calculate the summary statistics for the sequence lengths, image widths, and image heights
def get_sequence_shape(row):
    """Return the shape of the sequence in the format (W, H, N). The first two dimensions are interpreted as the width
    and height, respectively, since the images are to rotated 90 degrees to match the top-bottom fan orientations of the
    other datasets.
    """
    patient = f"patient{row['patient']:04d}"
    view = f"{row['view']}_half_sequence"
    image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}.nii.gz")

    return image_file.get_fdata().shape


metadata_df[["frame_width", "frame_height", "sequence_length"]] = metadata_df.apply(
    lambda row: get_sequence_shape(row), axis="columns", result_type="expand"
)

print(metadata_df["sequence_length"].describe())
print(metadata_df["frame_width"].describe())
print(metadata_df["frame_height"].describe())

# %%
# Plot the size of each image as a point on a scatter plot, coloured by image quality
fig, ax = plt.subplots()

symbols = {"Poor": "x", "Medium": "o", "Good": "s"}
for i, quality in enumerate(["Poor", "Medium", "Good"]):
    data = metadata_df[metadata_df["image_quality"] == quality]
    ax.scatter(
        data["frame_width"], data["frame_height"], marker=symbols[quality], label=quality, color=f"C{i}", alpha=0.5
    )

ax.set_xlabel("Frame width (pixels)")
ax.set_ylabel("Frame height (pixels)")
ax.set_aspect("equal")
ax.set_xlim(0, 1250)
ax.set_ylim(0, 1250)
ax.legend(loc="lower right")

plt.title("Image dimensions by quality")
plt.show()

x, y = metadata_df["frame_width"], metadata_df["frame_height"]
print("Image statistics:")
print(f"Width  | avg (+/- std): {np.mean(x):.2f} (+/- {np.std(x):.2f}), min: {np.min(x)}, max: {np.max(x)}")
print(f"Height | avg (+/- std): {np.mean(y):.2f} (+/- {np.std(y):.2f}), min: {np.min(y)}, max: {np.max(y)}")

# %%
# Verify image rotation using NumPy
example = metadata_df.iloc[0]
patient = f"patient{example['patient']:04d}"
view = f"{example['view']}_half_sequence"
image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}.nii.gz")
image = image_file.get_fdata()[..., 0]

# Rotate the image 90 degrees
rotated_image = np.rot90(image, axes=(1, 0))

plt.imshow(rotated_image, cmap="gray")
plt.axis("off")
plt.title("Rotated image")
plt.show()


# %%
# Tune scan mask segmentation
def segment_fan(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask


rows = 10
cols = 10

sample = metadata_df.sample(rows * cols, random_state=42)

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    example = sample.iloc[i]
    patient = f"patient{example['patient']:04d}"
    view = f"{example['view']}_half_sequence"
    image_file = nib.load(f"{DATA_DIR}/database_nifti/{patient}/{patient}_{view}.nii.gz")
    image = image_file.get_fdata()[..., 0]
    mask = segment_fan(image)
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
