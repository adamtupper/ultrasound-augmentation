"""Prepare the Butterfly dataset, splitting the training data into training and validation sets using a 8:2 split and
preserving the existing test split. The training and validation sets are split by patient to ensure that there is no
patient overlap between the splits.

Each example (a single image) is represented as an object in one of three JSON array files (`train.json`,
`validation.json`, or `test.json`). Each object has the following key/value pairs:
    - patient: The patient ID.
    - image: The path to the image file.
    - scan_mask: The path to the scan mask file.
    - class: The class name.
    - label: The integer label corresponding to the class.

Usage:
    python prepare_butterfly.py
        --dataset_dir /path/to/dataset
        --output_dir /path/to/output
        --version N
"""

import argparse
import glob
import json
import os
import shutil

import pandas as pd
import skimage
from sklearn.model_selection import GroupShuffleSplit

OUTPUT_NAME = "butterfly_v{}"
CLASS_TO_LABEL = {
    "carotid": 0,
    "2ch": 1,
    "lungs": 2,
    "ivc": 3,
    "4ch": 4,
    "bladder": 5,
    "thyroid": 6,
    "plax": 7,
    "morisons_pouch": 8,
}


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="prepare_butterfly.py")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="The path to the original dataset",
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
        help="The version number to assigne the processed dataset",
        required=True,
    )

    args = parser.parse_args()
    assert os.path.exists(args.dataset_dir), "dataset must be an existing directory"
    assert os.path.exists(args.output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))
    ), "a version of the dataset with this version number already exists"

    return args


def generate_scan_mask(output_dir: str, rel_image_path: str, rel_mask_path: str):
    """Generate a scan mask for an image using morphological operations.

    Args:
        output_dir (str): The output directory for the dataset.
        image_path (str): The path to the image file.
        mask_path (str): The path to save the scan mask file.
    """
    image = skimage.io.imread(rel_image_path)
    mask = image > 0  # Threshold the image
    mask = skimage.morphology.convex_hull_image(mask)  # Extract convex hull of the mask
    skimage.io.imsave(os.path.join(output_dir, rel_mask_path), mask.astype("uint8"), check_contrast=False)


def main():
    """Prepare the training, validation, and test sets for the Butterfly dataset."""
    args = parse_args()
    output_dir = os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))

    # Parse the metadata for each example
    examples = []
    for path in glob.glob(f"{args.dataset_dir}/*/*/*/*.png"):
        subpath = path.removeprefix(args.dataset_dir).removeprefix("/")
        subset, patient, label, filename = subpath.split("/")
        new_filename = f"{patient}_{label}_{filename}"

        examples.append(
            {
                "subset": "train" if "training" in subset else "test",
                "patient": int(patient),
                "class": label,
                "label": CLASS_TO_LABEL[label],
                "filename": new_filename,
                "filepath": path,
                "image": f"images/{new_filename}",
                "scan_mask": f"masks/scan/{new_filename}",
            }
        )
    df = pd.DataFrame.from_records(examples)

    # Create the scan masks
    mask_dir = os.path.join(output_dir, "masks", "scan")
    os.makedirs(mask_dir, exist_ok=True)
    df.apply(lambda x: generate_scan_mask(output_dir, x["filepath"], x["scan_mask"]), axis="columns")

    # Copy the images to the output directory
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    df.apply(lambda x: shutil.copy(x["filepath"], os.path.join(image_dir, x["filename"])), axis="columns")

    # Split the dataset into training, validation, and test sets
    test_df = df[df["subset"] == "test"]
    train_val_df = df[df["subset"] == "train"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(splitter.split(X=train_val_df, groups=train_val_df["patient"]))
    train_df = train_val_df.iloc[train_indices]
    val_df = train_val_df.iloc[val_indices]

    # Save the training, validation, and test indices to a JSON file
    for split, subset in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        subset = subset.drop(["filepath", "subset", "filename"], axis="columns").to_dict(orient="records")

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    # Save a file with the current date and commit hash
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"Version: {args.version}\n")
        f.write(f"Commit: {os.popen('git rev-parse HEAD').read().strip()}\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n")


if __name__ == "__main__":
    main()
