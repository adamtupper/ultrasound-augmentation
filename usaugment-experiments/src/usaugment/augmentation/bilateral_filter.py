"""A Random bilateral filter transform to reduce speckle noise as proposed in "Bilateral Filtering for Gray and Color
Images" (Tomasi & Manduchi, 1998). Uses the implementation from scikit-image.
"""

from typing import Any, Dict, Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from skimage.restoration import denoise_bilateral


class BilateralFilter(ImageOnlyTransform):
    """A Random bilateral filter transform to reduce speckle noise as proposed in "Bilateral Filtering for Gray and
    Color Images" (Tomasi & Manduchi, 1998). Uses the implementation from scikit-image.
    """

    def __init__(
        self,
        sigma_spatial: float | Tuple[float, float] = (0.1, 2.0),
        sigma_color: float | Tuple[float, float] = (0.0, 1.0),
        window_size: int = 5,
        p: float = 0.5,
    ) -> None:
        super(BilateralFilter, self).__init__(p=p)
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.window_size = window_size

    def apply(self, img: np.ndarray, **params: Any):
        img = img.copy()

        if isinstance(self.sigma_spatial, tuple) or isinstance(self.sigma_spatial, list):
            sigma_spatial = np.random.uniform(*self.sigma_spatial)
        else:
            sigma_spatial = self.sigma_spatial

        if isinstance(self.sigma_color, tuple) or isinstance(self.sigma_color, list):
            sigma_color = np.random.uniform(*self.sigma_color)
        else:
            sigma_color = self.sigma_color

        channel_axis = -1 if img.ndim == 3 else None
        denoised_img = denoise_bilateral(
            img,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            win_size=self.window_size,
            channel_axis=channel_axis,
        )

        if img.ndim == 2:
            # Single-channel image
            img = np.where(params["scan_mask"], denoised_img, img)
        else:
            # Multi-channel image
            img = np.where(params["scan_mask"][:, :, None], denoised_img, img)

        return img

    def get_params_dependent_on_data(self, params: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        return {"scan_mask": data["scan_mask"]}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import skimage

    image = skimage.io.imread("../../../figures/fatty_liver_v2_image_10_0.png")
    max_intensity = np.max(image)

    augmentation = BilateralFilter(sigma_spatial=1, sigma_color=0.5, p=1.0)

    augmented_image = augmentation.apply(img=image)

    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].set_title("Original Image")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Augmented Image")
    axes[1].imshow(augmented_image, cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()
