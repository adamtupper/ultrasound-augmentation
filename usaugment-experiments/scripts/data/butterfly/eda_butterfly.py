"""The Butterfly dataset was published on GitHub for the MIT Grand Hack 2018.

Dataset: https://github.com/ButterflyNetwork/MITGrandHack2018
"""

# %%
# Setup
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import skimage.io as io

ROOT_DIR = "/home-local2/adtup.extra.nobkp/project/data/raw/butterfly"

# %%
# Organize images into a dataframe
examples = []
for path in glob.glob(f"{ROOT_DIR}/*/*/*/*.png"):
    subpath = path.removeprefix(ROOT_DIR + "/")
    subset, patient, label, filename = subpath.split("/")
    examples.append(
        {
            "subset": "train" if "training" in subset else "test",
            "patient": int(patient),
            "label": label,
            "filename": filename,
            "filepath": path,
        }
    )
df = pd.DataFrame.from_records(examples)
df.head()

# %%
# Plot the number of examples per label, split by subset
ax = sns.countplot(data=df, x="label", hue="subset")
ax.set_title("Number of examples per class")
plt.xticks(rotation=45)
plt.show()

print(f"Total number of examples: {len(df)}")
print(f"Total number of train/validation examples: {len(df[df['subset'] == 'train'])}")
print(f"Total number of test examples: {len(df[df['subset'] == 'test'])}")

# %%
# Plot the number of examples per patient, split by label
fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.countplot(ax=ax, data=df, x="patient", hue="label")
ax.set_title("Number of examples per patient")
plt.show()

# %%
# Check the dimensions of the images
df["shape"] = df["filepath"].apply(lambda x: io.imread(x).shape)

print("Number of 3-channel images:", len(df[df["shape"].apply(lambda x: len(x) == 3)]))
print("Number of 1-channel images:", len(df[df["shape"].apply(lambda x: len(x) == 2)]))

# Plot the size of each image as a point on a scatterplot, coloured by class
fig, ax = plt.subplots()

labels = df["label"].unique()
markers = ["o", "x", "+", "s", "D", "v", "^", "<", ">", "p", "P", "*", "X"]
for label, marker in zip(labels, markers):
    label_df = df[df["label"] == label]
    ax.scatter(
        label_df["shape"].apply(lambda x: x[1]),
        label_df["shape"].apply(lambda x: x[0]),
        label=label,
        marker=marker,
        alpha=0.5,
    )
ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
ax.set_xlim(350, 550)
ax.set_ylim(350, 550)
ax.set_aspect("equal")
ax.set_title("Image dimensions by class")
ax.legend(ncol=1)

plt.show()

heights, widths = zip(*df["shape"])
print("Image statistics:")
print(
    f"Width  | avg (+/- std): {np.mean(widths):.2f} (+/- {np.std(widths):.2f}), min: {np.min(widths)}, max: {np.max(widths)}"
)
print(
    f"Height | avg (+/- std): {np.mean(heights):.2f} (+/- {np.std(heights):.2f}), min: {np.min(heights)}, max: {np.max(heights)}"
)

# %%
# Visualize a few images from each class
labels = df["label"].unique()

fig, axes = plt.subplots(4, 9, figsize=(12, 9))
for label, col in zip(labels, axes.transpose()):
    col[0].set_title(label)
    for i, ax in enumerate(col):
        label_df = df[df["label"] == label]
        filepath = label_df["filepath"].iloc[i]
        image = io.imread(filepath)
        ax.imshow(image, cmap="gray")
        ax.axis("off")

plt.tight_layout()
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

sample = df.sample(rows * cols, random_state=42)

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    image = io.imread(sample["filepath"].iloc[i])
    mask = segment_fan(image)
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
