"""Split the Fatty Liver dataset into training, validation, and test sets using a 7:1:2 split. The dataset
consists of 550 images collected from 55 patients (10 images per patient). The images are split by patient
to ensure that there is no patient overlap between the splits.

Each example (a single image) is represented as an object in one of three JSON array files (`train.json`,
`validation.json`, or `test.json`). Each object has the following key/value pairs:
    - patient: The patient ID.
    - image: The path to the image file.
    - steatosis: The percentage of hepatocytes with steatosis.
    - label: Whether or not the liver is fatty (0 if <5% hepatocytes with steatoisis, 1 otherwise).
    - pathology: 0 = Normal or 1 = NFLD.

Usage:
    python prepare_nfld.py
        --dataset /path/to/dataset
        --output_dir /path/to/output
        --version N
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy
import skimage
import skimage.io as io
from sklearn.model_selection import GroupShuffleSplit

OUTPUT_NAME = "fatty_liver_v{}"
LABEL_TO_CLASS = {0: "Normal", 1: "NFLD"}


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="prepare_fatty_liver.py")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The path to the original dataset (.mat file)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory in which to save the processed dataset",
        required=True,
    )
    parser.add_argument(
        "--version",
        type=int,
        help="The version number to assign the processed dataset",
        required=True,
    )

    args = parser.parse_args()
    assert os.path.isfile(args.dataset), "dataset must be an existing file"
    assert os.path.exists(args.output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))
    ), "a version of the dataset with this version number already exists"

    return args


def generate_scan_mask(image: np.ndarray):
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


def main():
    """Prepare the training, validation, and test sets for the Fatty Liver dataset."""
    args = parse_args()
    output_dir = os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))

    # Load the data
    data = scipy.io.loadmat(args.dataset)
    examples = data["data"][0]
    patient_ids = np.array([examples[0][0][0] for examples in examples])
    labels = np.array([examples[1][0][0] for examples in examples])
    steatosis_values = np.array([examples[2][0][0] for examples in examples])
    frame_sequences = [examples[3] for examples in examples]

    # Save each frame as a PNG image
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for id, frames in zip(patient_ids, frame_sequences):
        for i in range(frames.shape[0]):
            image_path = os.path.join(image_dir, f"{id}_{i}.png")
            io.imsave(image_path, frames[i])

    # Generate a scan mask for each image
    mask_dir = os.path.join(output_dir, "masks", "scan")
    os.makedirs(mask_dir, exist_ok=True)
    for id, frames in zip(patient_ids, frame_sequences):
        for i in range(frames.shape[0]):
            mask_path = os.path.join(mask_dir, f"{id}_{i}.png")
            mask = generate_scan_mask(frames[i])
            io.imsave(mask_path, mask, check_contrast=False)

    # Create a dataframe of examples
    examples = []
    for id, label, steatosis_value, frames in zip(
        patient_ids, labels, steatosis_values, frame_sequences
    ):
        for i in range(frames.shape[0]):
            examples.append(
                {
                    "patient": id,
                    "image": os.path.join("images", f"{id}_{i}.png"),
                    "scan_mask": os.path.join("masks", "scan", f"{id}_{i}.png"),
                    "steatosis": steatosis_value,
                    "label": label,
                    "pathology": LABEL_TO_CLASS[label],
                }
            )
    examples = pd.DataFrame.from_records(examples)

    # Split the dataset into training, validation, and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_val_indices, test_indices = next(
        splitter.split(X=examples, groups=examples["patient"])
    )
    train_val_examples = examples.iloc[train_val_indices]
    test_examples = examples.iloc[test_indices]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_indices, val_indices = next(
        splitter.split(X=train_val_examples, groups=train_val_examples["patient"])
    )
    train_examples = train_val_examples.iloc[train_indices]
    val_examples = train_val_examples.iloc[val_indices]

    # Save the training, validation, and test indices to a JSON file
    for split, subset in [
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ]:
        subset = subset.to_dict(orient="records")

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    # Save a file with the current date and commit hash
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"Version: {args.version}\n")
        f.write(f"Commit: {os.popen('git rev-parse HEAD').read().strip()}\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n")


if __name__ == "__main__":
    main()
