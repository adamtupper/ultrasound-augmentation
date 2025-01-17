# %%
# Setup

import matplotlib.pyplot as plt
import numpy as np
import skimage
from usaugment.augmentation.bilateral_filter import BilateralFilter
from usaugment.augmentation.depth_attenuation import DepthAttenuation
from usaugment.augmentation.gaussian_shadow import GaussianShadow
from usaugment.augmentation.haze_artifact import HazeArtifact

IMAGE_PATH = "../figures/fatty_liver_v2_image_10_0.png"
SCAN_MASK_PATH = "../figures/fatty_liver_v2_scan_mask_10_0.png"

image = skimage.io.imread(IMAGE_PATH) / 255.0
image = np.stack([image, image, image], axis=-1)
scan_mask = skimage.io.imread(SCAN_MASK_PATH)

# %%
# Depth Attenuation
attenuation_rate = 2.0
max_attenuation = 0.0
augmentation = DepthAttenuation(
    attenuation_rate=attenuation_rate, max_attenuation=max_attenuation, p=1.0
)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

axes[0].set_title("Original Image")
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)

axes[1].set_title("Attenuation Map")
axes[1].set_ylabel("✕", rotation=0, labelpad=0, fontsize=16)
axes[1].yaxis.set_label_coords(-0.1, 0.45)
axes[1].imshow(
    augmentation._generate_attenuation_map(*image.shape[:2], scan_mask=scan_mask)
)

axes[2].set_title("Augmented Image")
axes[2].set_ylabel("=", rotation=0, labelpad=0, fontsize=16)
axes[2].yaxis.set_label_coords(-0.1, 0.45)
axes[2].imshow(augmentation.apply(img=image, scan_mask=scan_mask))

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("../figures/depth_attenuation.pdf", bbox_inches="tight")
plt.show()

# %%
# Haze Artifact
radius = 0.5
sigma = 0.1
augmentation = HazeArtifact(radius=radius, sigma=sigma, p=1.0)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

axes[0].set_title("Original Image")
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)

axes[1].set_title("Haze Image")
axes[1].set_ylabel("+", rotation=0, labelpad=0, fontsize=16)
axes[1].yaxis.set_label_coords(-0.1, 0.45)
haze_image = augmentation._generate_haze(*image.shape[:2])
haze_image = haze_image * scan_mask
axes[1].imshow(haze_image)

axes[2].set_title("Augmented Image")
axes[2].set_ylabel("=", rotation=0, labelpad=0, fontsize=16)
axes[2].yaxis.set_label_coords(-0.1, 0.45)
axes[2].imshow(augmentation.apply(img=image, scan_mask=scan_mask))

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("../figures/haze_artifact.pdf", bbox_inches="tight")
plt.show()

# %%
# Gaussian Shadow
seed = 5
strength = 0.8
sigma_x = 0.1
sigma_y = 0.1
augmentation = GaussianShadow(
    strength=strength, sigma_x=sigma_x, sigma_y=sigma_y, p=1.0
)
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

axes[0].set_title("Original Image")
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)

axes[1].set_title("Shadow Image")
axes[1].set_ylabel("✕", rotation=0, labelpad=0, fontsize=16)
axes[1].yaxis.set_label_coords(-0.1, 0.45)
np.random.seed(seed)
shadow_image = augmentation._generate_shadow_image(
    image.shape[0], image.shape[1], scan_mask=scan_mask
)
axes[1].imshow(shadow_image)

axes[2].set_title("Augmented Image")
axes[2].set_ylabel("=", rotation=0, labelpad=0, fontsize=16)
axes[2].yaxis.set_label_coords(-0.1, 0.45)
np.random.seed(seed)
augmented_image = augmentation.apply(img=image, scan_mask=scan_mask)
axes[2].imshow(augmented_image)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("../figures/gaussian_shadow.pdf", bbox_inches="tight")
plt.show()

# %%
# Bilateral Filter
sigma_spatial = 0.5
sigma_color = 0.5
augmentation = BilateralFilter(
    sigma_spatial=sigma_spatial, sigma_color=sigma_color, window_size=5, p=1.0
)

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

axes[0].set_title("Original Image")
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)

axes[1].set_title("Augmented Image")
augmented_image = augmentation.apply(img=image, scan_mask=scan_mask)
axes[1].imshow(augmented_image)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("../figures/bilateral_filter.pdf", bbox_inches="tight")
plt.show()

# %%
