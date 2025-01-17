import albumentations as A

from usaugment.augmentation import BilateralFilter, DepthAttenuation, GaussianShadow, HazeArtifact

TRANSFORMS_DICT = {
    "bilateral_filter": BilateralFilter(sigma_spatial=[0.05, 1.0], sigma_color=[0.05, 1.0], window_size=5, p=1.0),
    "brightness": A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=[0.0, 0.0], p=1.0),
    "contrast": A.RandomBrightnessContrast(brightness_limit=[0.0, 0.0], contrast_limit=[-0.2, 0.2], p=1.0),
    "depth_attenuation": DepthAttenuation(attenuation_rate=[0.0, 3.0], max_attenuation=0.0, p=1.0),
    "flip_horizontal": A.HorizontalFlip(p=1.0),
    "flip_vertical": A.VerticalFlip(p=1.0),
    "gamma": A.RandomGamma(gamma_limit=[80, 120], p=1.0),
    "gaussian_noise": A.GaussNoise(var_limit=0.0225, mean=0, per_channel=False, noise_scale_factor=1.0, p=1.0),
    "gaussian_shadow": GaussianShadow(strength=[0.25, 0.8], sigma_x=[0.01, 0.2], sigma_y=[0.01, 0.2], p=1.0),
    "haze_artifact": HazeArtifact(radius=[0.05, 0.95], sigma=[0.0, 0.1], p=1.0),
    "identity": A.NoOp(p=1.0),
    "random_crop": A.RandomCrop(width=224, height=224, p=1.0),
    "rotate": A.Rotate(limit=[-30, 30], border_mode=0, value=0, p=1.0),
    "translate": A.ShiftScaleRotate(
        shift_limit=[-0.0625, 0.0625],
        scale_limit=[0.0, 0.0],
        rotate_limit=[0, 0],
        interpolation=1,
        border_mode=0,
        value=0,
        p=1.0,
    ),
    "zoom": A.ShiftScaleRotate(
        shift_limit=[0.0, 0.0],
        scale_limit=[-0.1, 0.1],
        rotate_limit=[0, 0],
        interpolation=1,
        border_mode=0,
        value=0,
        p=1.0,
    ),
}


def parse_transforms(transforms_string):
    transform_names = transforms_string.strip().split(",")

    return [TRANSFORMS_DICT[transform] for transform in transform_names]


def configure_trivial_augment(config):
    """Configure TrivialAugment."""
    transforms = parse_transforms(config.transforms)

    return A.Compose(transforms=[A.OneOf(p=1.0, transforms=transforms) for _ in range(config.num_operations)])
