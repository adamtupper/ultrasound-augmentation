"""The dataset used and described in "Transfer learning with deep convolutional neural network for liver steatosis
assessment in ultrasound images" (Byra et al., 2018)

Dataset: https://zenodo.org/records/1009146
Paper: https://link.springer.com/article/10.1007/s11548-018-1843-2
"""

# %%
# Setup
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import skimage

DATA_PATH = "/home-local2/adtup.extra.nobkp/project/data/raw/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"

# %%
# Load data
data = scipy.io.loadmat(DATA_PATH)
print(data.keys())

examples = data["data"][0]

# %%
# Verify the number of patients (should be 55)
print(len(examples))

# %%
index = np.array([examples[0][0][0] for examples in examples])
label = np.array([examples[1][0][0] for examples in examples])
seatosis = np.array([examples[2][0][0] for examples in examples])
frames = [examples[3] for examples in examples]

# %%
# Plot the distribution of steatosis levels
plt.hist(seatosis, bins=np.arange(0, 105, 5), alpha=0.8)
plt.xlim(0, 100)
plt.ylim(0, 20)
plt.xticks(np.arange(0, 101, 20))
plt.xticks(np.arange(0, 101, 5), minor=True)
plt.yticks(np.arange(0, 21, 5))
plt.yticks(np.arange(0, 21, 1), minor=True)
plt.gca().set_axisbelow(True)
plt.grid(which="major", lw=0.5)
plt.grid(which="minor", lw=0.25)
plt.xlabel("Steatosis (%)")
plt.ylabel("Count")
plt.title("Steatosis level across the population of patients")
plt.show()

# %%
# Verify that the steatosis level is consistent with the label (there should be 38 patients with steatosis >= 5)
print("No. of patients with a steatosis level >= 5:", np.sum(seatosis[label.astype(bool)] >= 5))

# %%
# Plot the class distribution
plt.bar(["Normal", "NFLD"], [np.sum(label == 0), np.sum(label == 1)], alpha=0.8)
plt.text(0, np.sum(label == 0) + 1, np.sum(label == 0), ha="center")
plt.text(1, np.sum(label == 1) + 1, np.sum(label == 1), ha="center")
plt.ylim(0, 42.5)
plt.ylabel("Count")
plt.gca().set_axisbelow(True)
plt.grid(which="major", axis="y", lw=0.5, zorder=3)
plt.title("Class distribution")
plt.show()

# %%
# Visualize the frame sequence for the first patient
patient = 0

fig, axes = plt.subplots(1, 10, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(frames[patient][i], cmap="gray")
    ax.axis("off")

# %%
# Inspect the pixel values of the first frame sequence
pixel_values = []
for frame in frames[0]:
    pixel_values.append(frame.flatten())

pixel_values = np.array(pixel_values)
print(np.unique(pixel_values))


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

    # Remove remaining small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)

    # Dilate the mask
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


rows = 5
cols = 11

# Visualize the scan mask of the first frame for each patient
subset = [frames[i][0] for i in range(len(frames))]

fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    image = subset[i]
    if image.ndim == 3:
        image = image.mean(axis=-1)
    mask = segment_fan(image)
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.4, vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
# Verify that all the images are single-channel
for i in range(len(frames)):
    for j in range(len(frames[i])):
        assert frames[i][j].ndim == 2

# %%
