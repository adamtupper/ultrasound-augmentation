# %%
# Setup

import matplotlib.pyplot as plt
import pandas as pd

# %%
# Load spreadsheet
df = pd.read_excel(
    "../Full-Text Ultrasound Articles Review.xlsx"
)
df.drop(columns=["Publication Title", "DOI", "URL", "Conference Name"], inplace=True)
df.rename(
    columns={
        "Publication Year": "publication_year",
        "Title": "title",
        "Author": "author",
        "Decision": "decision",
        "Exclusion Reason": "exclusion_reason",
        "Augmentations": "augmentations",
        "Datasets": "datasets",
        "Task": "task",
    },
    inplace=True,
)
df["augmentations"] = df["augmentations"].astype("string")
df["augmentations"] = df["augmentations"].fillna("none")
df = df[df["decision"] == "Include"]


print("Number of included articles:", len(df))
df.head()

# %%
# Merge duplicates by assigning each augmentation a canonical name
augmentations = sorted(
    df["augmentations"].str.split("; ", expand=True).stack().unique()
)

name_to_canonical_name = {
    "ClassMix": "classmix",
    "Cone random position augmentation": "cone_random_position_augmentation",
    "CutMix": "cutmix",
    "CutOut": "cutout",
    "Gaussian blur": "gaussian_blur",
    "Gaussian noise": "gaussian_noise",
    "Gaussian shadowing": "gaussian_shadowing_ostvik2021",
    "Gaussian shadows": "gaussian_shadowing_ostvik2021",
    "Horizontal flip": "flip_horizontal",
    "IPM": "image_puzzle_mixing_ding2024",
    "JPEG compression": "jpeg_compression",
    "Mixup": "mixup",
    "Mixup manifold": "manifold_mixup",
    "Myocardium intensity augmentation": "myocardium_intensity_augmentation_sfakianakis2023",
    "Noise": "unspecified_random_noise",
    "Perspective random position augmentations": "perspective_random_position_augmentation_sfakianakis2023",
    "Random intensity windowing": "random_intensity_windowing",
    "Resize": "resize",
    "Rotation": "rotation",
    "Scaling": "resize",
    "Shear": "shear",
    "Speckle": "speckle_parameter_map_augmentation_singla2022",
    "TGC": "time_gain_compensation_singla2022",
    "Unspecified augmentation": "unspecified_augmentation",
    "Zoom (US specific)": "fan_shape_preserving_zoom_singla2022",
    "acoustic shadow": "acoustic_shadow_singla2022",
    "bilateral filtering": "bilateral_filtering",
    "blurring": "blur",
    "brightness": "brightness_adjustment",
    "brightness adjustment": "brightness_adjustment",
    "center crop": "center_crop",
    "color jitter": "color_jitter",
    "color jittering": "color_jitter",
    "contrast": "contrast_adjustment",
    "contrast adjustment": "contrast_adjustment",
    "contrast enhancement": "adaptive_gamma_correction_rahman2016",
    "crop": "random_crop",
    "cropping": "random_crop",
    "cutout": "cutout",
    "depth attenuation": "depth_attenuation_ostvik2021",
    "diagonal flip": "flip_diagonal",
    "disentanglement and fusion augmentation (DF-Aug)": "disentanglement_and_fusion_augmentation_monkam2022a",
    "elastic deformation": "elastic_transform",
    "elastic transform": "elastic_transform",
    "field-of-view masking": "field_of_view_masking_pasdeloup2023",
    "fine detail enhancement": "unsharp_masking_monkam2023a",
    "flipping": "flip_unspecified",
    "frame jittering": "jitter",
    "frame skipping": "frame_skipping",
    "frequency-domain interpolation": "frequency_domain_interpolation_ding2024",
    "gamma adjustment": "gamma_adjustment",
    "gamma correcting": "gamma_adjustment",
    "gamma intensity transform": "gamma_adjustment",
    "gamma intensity transformation": "gamma_adjustment",
    "gamma transformation": "gamma_adjustment",
    "gamma transformations": "gamma_adjustment",
    "gaussian blur": "gaussian_blur",
    "geometric transforms": "unspecified_geometric_transforms",
    "gray scale": "grayscale_conversion",
    "grayscale linear transformations": "grayscale_conversion",
    "haze artifact application": "haze_artifacts_ostvik2021",
    "haze artifacts": "haze_artifacts_ostvik2021",
    "horizontal flip": "flip_horizontal",
    "horizontal translation": "translation_horizontal",
    "hue adjustment": "hue_adjustment",
    "image resampling artifacts": "image_resampling_artifacts",
    "image synthesis using deep diffusion model and GAN": "image_synthesis",
    "intensity adjustment": "intensity_adjustment",
    "intensity scaling": "intensity_adjustment",
    "intensity windowing": "random_intensity_windowing",
    "mirroring": "mirror",
    "multi-level speckle noise addition": "multi_level_speckle_noise_monkam2023a",
    "multiplicative noise": "multiplicative_noise",
    "multiplicative noise addition": "multiplicative_noise",
    "noise": "unspecified_random_noise",
    "none": "none",
    "non-linear colour maps": "nonlinear_colour_maps_pasdeloup2023",
    "normalization": "normalization",
    "occlusion strategies": "unspecified_occlusion",
    "piecewise affine transform": "piecewise_affine_transform",
    "pixel noise": "unspecified_random_noise",
    "projective transform": "projective_transform",
    "random crop": "random_crop",
    "random erasing": "random_erasing",
    "random noise": "unspecified_random_noise",
    "random occluding": "unspecified_occlusion",
    "random resized crop": "random_resized_crop",
    "resize": "resize",
    "resized crop": "random_resized_crop",
    "resizing": "resize",
    "rotation": "rotation",
    "salt and pepper noise": "salt_and_pepper_noise",
    "saturation adjustment": "saturation_adjustment",
    "scale-crop": "random_resized_crop",
    "scaling": "resize",
    "shadowing": "gaussian_shadowing_ostvik2021",
    "sharpening": "sharpen",
    "shear": "shear",
    "simulation of low resolution": "low_resolution_simulation_isensee2021",
    "speckle noise addition": "speckle_noise_addition_wang2022d",
    "speckle noise suppression": "speckle_reducing_anisotropic_diffusion_yu2002",
    "speckle reduction": "bilateral_filtering",
    "temporal reversing": "temporal_reversing",
    "translation": "translation_unspecified",
    "unspecified": "unspecified_augmentation",
    "unspecified flip": "flip_unspecified",
    "unspecified flipping": "flip_unspecified",
    "unspecified translation": "translation_unspecified",
    "vertical flip": "flip_vertical",
    "vertical translation": "translation_vertical",
    "wrapping": "wrap",
    "zoom": "resize",
}

# Check that each augmentation is assigned to a canonical name
for augmentation in augmentations:
    if augmentation not in name_to_canonical_name:
        print(augmentation)

print("Number of unique augmentations:", len(set(name_to_canonical_name.values())))


# %%
# Map each augmentation to their canonical names
def map_to_canonical(augmentations):
    canonical_augmentations = []
    for aug in augmentations:
        canonical_name = name_to_canonical_name.get(aug)
        if canonical_name is not None:
            canonical_augmentations.append(canonical_name)
        else:
            raise ValueError(f"'{aug}' does not have a canonical name")

    return canonical_augmentations


df["canonical_augmentations"] = df["augmentations"].apply(
    lambda x: ";".join(map_to_canonical(x.split("; ")))
)

# %%
# Count the number of uses of each augmentation
augmentation_counts = (
    df["canonical_augmentations"].str.split(";", expand=True).stack().value_counts()
)

for name, count in sorted(
    augmentation_counts.items(), key=lambda x: x[1], reverse=True
):
    print(f"{name}: {count}")

# %%
# Group similar augmentations (e.g., all flips are grouped together)
canonical_name_to_group = {
    "flip_horizontal": "flip",
    "flip_diagonal": "flip",
    "flip_unspecified": "flip",
    "flip_vertical": "flip",
    "translation_horizontal": "translation",
    "random_crop": "random_crop",
    "random_resized_crop": "random_crop",
    "translation_unspecified": "translation",
    "translation_vertical": "translation",
    "jitter": "translation",
}


# Map each augmentation to their canonical names
def canonical_to_group(augmentations):
    grouped_augmentations = []
    for aug in augmentations:
        if aug in canonical_name_to_group:
            group = canonical_name_to_group.get(aug)
        else:
            group = aug

        grouped_augmentations.append(group)

    return grouped_augmentations


df["grouped_augmentations"] = df["canonical_augmentations"].apply(
    lambda x: ";".join(canonical_to_group(x.split(";")))
)

print(
    "Number of unique augmentation groups:",
    len(set(df["grouped_augmentations"].str.split(";").explode())),
)


# %%
# Assign display names to each augmentation group
def group_to_display_name(augmentations):
    display_names = []
    for aug in augmentations:
        group = display_name_map.get(aug, None)

        if group is not None:
            display_names.append(group)

    return display_names


display_name_map = {
    "acoustic_shadow_singla2022": "Acoustic shadow (Singla et al., 2022)",
    "adaptive_gamma_correction_rahman2016": "Adaptive gamma correction (Rahman et al., 2016)",
    "bilateral_filtering": "Bilateral filtering",
    "blur": "Blur",
    "brightness_adjustment": "Brightness adj.",
    "center_crop": "Center crop",
    "classmix": "ClassMix",
    "color_jitter": "Color jitter",
    "cone_random_position_augmentation": "Cone position augmentation",
    "contrast_adjustment": "Contrast adj.",
    "cutmix": "CutMix",
    "cutout": "Cutout",
    "depth_attenuation_ostvik2021": "Depth attenuation (Ostvik et al., 2021)",
    "elastic_transform": "Elastic transform",
    "fan_shape_preserving_zoom_singla2022": "Fan shape preserving zoom (Singla et al., 2022)",
    "field_of_view_masking_pasdeloup2023": "Field-of-view masking (Pasdeloup et al., 2023)",
    "flip": "Flip",
    "frequency_domain_interpolation_ding2024": "Frequency domain mixing (Ding et al., 2024)",
    "gamma_adjustment": "Gamma adj.",
    "gaussian_noise": "Gaussian noise",
    "gaussian_shadowing_ostvik2021": "Gaussian shadowing (Ostvik et al., 2021)",
    "haze_artifacts_ostvik2021": "Haze artifacts (Ostvik et al., 2021)",
    "hue_adjustment": "Hue adj.",
    "image_puzzle_mixing_ding2024": "Image puzzle mixing (Ding et al., 2024)",
    "image_resampling_artifacts": "Image resampling artifacts",
    "intensity_adjustment": "Intensity adj.",
    "jpeg_compression": "JPEG compression",
    "low_resolution_simulation_isensee2021": "Low resolution simulation (Isensee et al., 2021)",
    "manifold_mixup": "Manifold mixup",
    "mirror": "Mirror",
    "mixup": "Mixup",
    "multi_level_speckle_noise_monkam2023a": "Multi-level speckle noise (Monkam et al., 2023)",
    "multiplicative_noise": "Multiplicative noise",
    "myocardium_intensity_augmentation_sfakianakis2023": "Myocardium intensity augmentation (Sfakianakis et al., 2023)",
    "nonlinear_colour_maps_pasdeloup2023": "Non-linear colour map (Pasdeloup et al., 2023)",
    "normalization": "Normalization",
    "perspective_random_position_augmentation_sfakianakis2023": "Perspective position augmentation (Sfakianakis et al., 2023)",
    "random_erasing": "Random erasing",
    "random_intensity_windowing": "Intensity windowing",
    "random_crop": "Random crop",
    "resize": "Resize",
    "rotation": "Rotation",
    "salt_and_pepper_noise": "Salt and pepper noise",
    "sharpen": "Sharpen",
    "shear": "Shear",
    "speckle_noise_addition_wang2022d": "Speckle noise (Wang et al., 2022)",
    "speckle_parameter_map_augmentation_singla2022": "Speckle parameter map (Singla et al., 2022)",
    "speckle_reducing_anisotropic_diffusion_yu2002": "Speckle reduction (Yu et al., 2002)",
    "time_gain_compensation_singla2022": "Time gain compensation (Singla et al., 2022)",
    "translation": "Translation",
    "unsharp_masking_monkam2023a": "Unsharp masking (Monkam et al., 2023)",
    "unspecified_occlusion": "Occlusion",
    "unspecified_random_noise": "Random noise",
    "wrap": "Wrap",
}

df["display_name"] = df["grouped_augmentations"].apply(
    lambda x: ";".join(group_to_display_name(x.split(";")))
)


# %%
# Assign short display names to each augmentation group
def group_to_short_display_name(augmentations):
    display_names = []
    for aug in augmentations:
        group = short_display_name_map.get(aug, None)

        if group is not None:
            display_names.append(group)

    return display_names


short_display_name_map = {
    "acoustic_shadow_singla2022": "Acoustic shadow",
    "adaptive_gamma_correction_rahman2016": "Adaptive gamma correction",
    "bilateral_filtering": "Bilateral filtering",
    "blur": "Blur",
    "brightness_adjustment": "Brightness adj.",
    "center_crop": "Center crop",
    "classmix": "ClassMix",
    "color_jitter": "Color jitter",
    "cone_random_position_augmentation": "Cone position adj.",
    "contrast_adjustment": "Contrast adj.",
    "cutmix": "CutMix",
    "cutout": "Cutout",
    "depth_attenuation_ostvik2021": "Depth attenuation",
    "elastic_transform": "Elastic transform",
    "fan_shape_preserving_zoom_singla2022": "Fan-shape preserving zoom",
    "field_of_view_masking_pasdeloup2023": "Field-of-view masking",
    "flip": "Flip",
    "frequency_domain_interpolation_ding2024": "Frequency domain mixing",
    "gamma_adjustment": "Gamma adj.",
    "gaussian_noise": "Gaussian noise",
    "gaussian_shadowing_ostvik2021": "Gaussian shadow",
    "haze_artifacts_ostvik2021": "Haze artifact addition",
    "hue_adjustment": "Hue adj.",
    "image_puzzle_mixing_ding2024": "Image puzzle mixing",
    "image_resampling_artifacts": "Image resampling artifacts",
    "intensity_adjustment": "Intensity adj.",
    "jpeg_compression": "JPEG compression",
    "low_resolution_simulation_isensee2021": "Low resolution simulation",
    "manifold_mixup": "Manifold mixup",
    "mirror": "Mirror",
    "mixup": "Mixup",
    "multi_level_speckle_noise_monkam2023a": "Multi-level speckle noise",
    "multiplicative_noise": "Multiplicative noise",
    "myocardium_intensity_augmentation_sfakianakis2023": "Myocardium intensity adj.",
    "nonlinear_colour_maps_pasdeloup2023": "Nonlinear colour map",
    "normalization": "Normalization",
    "perspective_random_position_augmentation_sfakianakis2023": "Perspective position adj.",
    "random_erasing": "Random erasing",
    "random_intensity_windowing": "Intensity windowing",
    "random_crop": "Random crop",
    "resize": "Resize",
    "rotation": "Rotation",
    "salt_and_pepper_noise": "Salt and pepper noise",
    "sharpen": "Sharpen",
    "shear": "Shear",
    "speckle_noise_addition_wang2022d": "Speckle noise",
    "speckle_parameter_map_augmentation_singla2022": "Speckle parameter map",
    "speckle_reducing_anisotropic_diffusion_yu2002": "Speckle reduction",
    "time_gain_compensation_singla2022": "Time gain compensation",
    "translation": "Translation",
    "unsharp_masking_monkam2023a": "Unsharp masking",
    "unspecified_occlusion": "Occlusion",
    "unspecified_random_noise": "Random noise",
    "wrap": "Wrap",
}

df["short_display_name"] = df["grouped_augmentations"].apply(
    lambda x: ";".join(group_to_short_display_name(x.split(";")))
)

# %%
# Create a lollipop plot of the number of uses of each augmentation (group)
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

counts = (
    df[df["display_name"] != ""]["display_name"]
    .str.split(";", expand=True)
    .stack()
    .value_counts()
    .rename_axis("augmentation")
    .reset_index(name="usage_count")
)

ordered_df = counts.sort_values(by="usage_count")
my_range = range(1, len(counts.index) + 1)

# The horizontal plot is made using the hline function
fig, ax = plt.subplots(figsize=(6.5, 12))
plt.hlines(y=my_range, xmin=0, xmax=ordered_df["usage_count"], color="skyblue")
plt.plot(ordered_df["usage_count"], my_range, "o")

# Add labels to the right of each bar
for i, (augmentation, usage_count) in enumerate(
    zip(ordered_df["augmentation"], ordered_df["usage_count"])
):
    ax.text(usage_count + 2, i + 1, str(augmentation), ha="left", va="center")

# Add titles and axis names
# plt.yticks(my_range, ordered_df["augmentation"])
plt.title("Data Augmentation Counts Among Reviewed Articles", loc="left")
plt.xlabel("Number of uses")
plt.ylabel("Augmentation")

# Show the plot
plt.show()

# %%
US_SPECIFIC_AUGMENTATIONS = short_display_name_map = {
    "Acoustic shadow",
    "Bilateral filtering",
    "Cone position adj.",
    "Depth attenuation",
    "Fan-shape preserving zoom",
    "Field-of-view masking",
    "Frequency domain mixing",
    "Gaussian shadow",
    "Haze artifact addition",
    "Multi-level speckle noise",
    "Myocardium intensity adj.",
    "Nonlinear colour map",
    "Perspective position adj.",
    "Speckle noise",
    "Speckle parameter map",
    "Speckle reduction",
    "Time gain compensation",
}

fig, ax = plt.subplots(figsize=(12, 4.5))

counts = (
    df[df["short_display_name"] != ""]["short_display_name"]
    .str.split(";", expand=True)
    .stack()
    .value_counts()
    .rename_axis("augmentation")
    .reset_index(name="usage_count")
)

counts["ultrasound_specific"] = counts["augmentation"].apply(
    lambda x: x in US_SPECIFIC_AUGMENTATIONS
)

ordered_df = counts.sort_values(by="usage_count").reset_index()
my_range = range(0, len(counts.index))

# Make the plot
(
    markerline,
    stemline,
    baseline,
) = ax.stem(
    ordered_df[~ordered_df["ultrasound_specific"]].index,
    ordered_df[~ordered_df["ultrasound_specific"]]["usage_count"],
    basefmt=" ",
    markerfmt="o",
    label="General",
)
plt.setp(stemline, linewidth=1.25)
plt.setp(markerline, markersize=5)

(
    markerline,
    stemline,
    baseline,
) = ax.stem(
    ordered_df[ordered_df["ultrasound_specific"]].index,
    ordered_df[ordered_df["ultrasound_specific"]]["usage_count"],
    basefmt=" ",
    markerfmt="D",
    linefmt="C1",
    label="Ultrasound-specific",
)
plt.setp(stemline, linewidth=1.25)
plt.setp(markerline, markersize=5)

plt.grid(axis="y", linestyle="--", alpha=0.6, which="major")
plt.grid(visible=False, axis="x", which="both")

plt.legend()

plt.xticks(my_range, ordered_df["augmentation"], rotation=45, ha="right")

plt.ylabel("Number of articles")
plt.yticks(range(0, 65, 5))
plt.ylim(0, 65)

plt.tight_layout()
plt.savefig("../figures/augmentation_use.png")
plt.savefig("../figures/augmentation_use.pdf")
plt.show()

# %%
# Create a pie chart of the augmentation counts
augmentation_counts = (
    df["grouped_augmentations"].str.split(";", expand=True).stack().value_counts()
)

plt.figure(figsize=(15, 12))
plt.pie(augmentation_counts, labels=augmentation_counts.index, autopct="%1.1f%%")
plt.show()
# %%
# Create a bar chart of the augmentation counts
plt.figure(figsize=(12, 12))

augmentation_counts.plot(kind="barh")

# Add the counts to the bars
for i, count in enumerate(augmentation_counts):
    plt.text(count, i, str(count))

plt.show()
# %%
df.head()


# %%
def contains_augmentation(augmentation, augmentations):
    return augmentation in augmentations.split(";")


# For every unique augmentation in canonical_augmentations, create a new column with a boolean value indicating whether the augmentation is present in the list of augmentations
for augmentation in augmentation_counts.index:
    df[augmentation] = df["canonical_augmentations"].apply(
        lambda x: contains_augmentation(augmentation, x)
    )

# %%
# Count the number of rows which use no augmentations (i.e., none is the only augmentation)
no_augmentations = df["none"].sum()
print(f"Number of articles with no augmentations: {no_augmentations}")

# %%
df.columns
# %%
df["saturation_adjustment"].sum()


# %%
# Count the number of augmentations used in each article
def count_augmentations(augmentations):
    if augmentations == "none":
        return 0
    else:
        return len(augmentations.split(";"))


df["num_augmentations"] = df["grouped_augmentations"].apply(count_augmentations)


# %%
# Number of augmentations used in each article
print(df["num_augmentations"].value_counts())

# Plot a pie chart of the number of augmentations used in the articles
num_augmentations_counts = df["num_augmentations"].value_counts()
plt.figure(figsize=(12, 12))

plt.pie(
    num_augmentations_counts, labels=num_augmentations_counts.index, autopct="%1.1f%%"
)
plt.title("Number of augmentations used in each article")
plt.show()

# %%
# Plot the percentage of articles that use data augmentation by publication year
articles_per_year = df.groupby("publication_year").size()
none_per_year = df[df["none"]].groupby("publication_year").size()
rotation_per_year = df[df["rotation"]].groupby("publication_year").size()
horizontal_flip_per_year = df[df["flip_horizontal"]].groupby("publication_year").size()
resize_per_year = df[df["resize"]].groupby("publication_year").size()
random_crop_per_year = df[df["random_crop"]].groupby("publication_year").size()
vertical_flip_per_year = df[df["flip_vertical"]].groupby("publication_year").size()
translation_per_year = (
    df[
        df["translation_unspecified"]
        | df["translation_horizontal"]
        | df["translation_vertical"]
    ]
    .groupby("publication_year")
    .size()
)
contrast_adjustment_per_year = (
    df[df["contrast_adjustment"]].groupby("publication_year").size()
)
brightness_adjustment_per_year = (
    df[df["brightness_adjustment"]].groupby("publication_year").size()
)
gaussian_noise_per_year = df[df["gaussian_noise"]].groupby("publication_year").size()
gamma_adjustment_per_year = (
    df[df["gamma_adjustment"]].groupby("publication_year").size()
)

none_per_year.plot(kind="line", figsize=(12, 12), label="None")
rotation_per_year.plot(kind="line", figsize=(12, 12), label="Rotation")
horizontal_flip_per_year.plot(kind="line", figsize=(12, 12), label="Horizontal flip")
resize_per_year.plot(kind="line", figsize=(12, 12), label="Resize")
random_crop_per_year.plot(kind="line", figsize=(12, 12), label="Random crop")
vertical_flip_per_year.plot(kind="line", figsize=(12, 12), label="Vertical flip")
translation_per_year.plot(kind="line", figsize=(12, 12), label="Translation")
contrast_adjustment_per_year.plot(
    kind="line", figsize=(12, 12), label="Contrast adjustment"
)
brightness_adjustment_per_year.plot(
    kind="line", figsize=(12, 12), label="Brightness adjustment"
)
gaussian_noise_per_year.plot(kind="line", figsize=(12, 12), label="Gaussian noise")
gamma_adjustment_per_year.plot(kind="line", figsize=(12, 12), label="Gamma adjustment")


plt.legend()
plt.show()


# %%
