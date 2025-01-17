"""An implementation of the haze artifact application transform described in "Myocardial Function Imaging in
Echocardiography Using Deep Learning" (Ostvik et al., 2021).
"""

from typing import Any, Dict, Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class HazeArtifact(ImageOnlyTransform):
    """
    An implementation of the haze artifact application transform described in "Myocardial Function Imaging in
    Echocardiography Using Deep Learning" (Ostvik et al., 2021).
    """

    def __init__(
        self,
        radius: float | Tuple[float, float] = (0.05, 0.95),
        sigma: float | Tuple[float, float] = (0, 0.1),
        p: float = 0.5,
    ) -> None:
        super(HazeArtifact, self).__init__(p=p)
        self.radius = radius
        self.sigma = sigma

    def apply(self, img: np.ndarray, **params: Any):
        img = img.copy()

        haze = self._generate_haze(width=img.shape[1], height=img.shape[0])
        haze = haze * params["scan_mask"]

        if img.ndim == 2:
            # Single-channel image
            img = img + 0.5 * haze.astype(img.dtype)
        else:
            # Multi-channel image
            img = img + 0.5 * haze[:, :, None].astype(img.dtype)

        # Clip the image to [0, 1]
        img = np.clip(img, 0, 1)

        return img

    def get_params_dependent_on_data(self, params: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        return {"scan_mask": data["scan_mask"]}

    def _generate_haze(
        self,
        height,
        width,
    ):
        """Generate a haze artifact for the image."""
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)

        r = np.sqrt((xv - 0.5) ** 2 + (yv - 0) ** 2)

        if isinstance(self.radius, tuple) or isinstance(self.radius, list):
            haze_radius = np.random.uniform(*self.radius)
        else:
            haze_radius = self.radius

        if isinstance(self.sigma, tuple) or isinstance(self.sigma, list):
            haze_sigma = np.random.uniform(*self.sigma)
        else:
            haze_sigma = self.sigma

        haze = np.random.uniform(0, 1, (height, width))
        haze *= np.exp(-((r - haze_radius) ** 2) / (2 * haze_sigma**2))

        return haze


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import skimage

    image = skimage.io.imread("../../../figures/fatty_liver_v2_image_10_0.png")
    max_intensity = np.max(image)

    augmentation = HazeArtifact(radius=0.5, sigma=0.1, p=1.0)

    augmented_image = augmentation.apply(img=image)
    attenuation_map = augmentation._generate_haze(*image.shape[:2])

    fig, axes = plt.subplots(1, 3, sharey=True)
    axes[0].set_title("Original Image")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=max_intensity)
    axes[1].set_title("Haze Artifact")
    axes[1].imshow(attenuation_map, vmin=0, vmax=1)
    axes[2].set_title("Augmented Image")
    axes[2].imshow(augmented_image, cmap="gray", vmin=0, vmax=max_intensity)
    plt.tight_layout()
    plt.show()
