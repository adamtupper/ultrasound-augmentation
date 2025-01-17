# %%
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

RESULTS_DIR = "../outputs/results/"
TASKS = {
    "aul_liver_segmentation": "aul_liver_v5_liver_segmentation",
    "aul_mass_segmentation": "aul_mass_v5_mass_segmentation",
    "camus_segmentation": "camus_v4_segmentation",
    "mmotu_segmentation": "mmotu_v2_segmentation",
    "open_kidney_capsule_segmentation": "open_kidney_v3_capsule_segmentation",
    "psfhs_segmentation": "psfhs_v2_segmentation",
    "stanford_thyroid_segmentation": "stanford_thyroid_v4_segmentation",
}

# %%
# %%
# Combine the results on each task into a a single data frame
for i, task in enumerate(TASKS):
    results = pd.read_csv(
        os.path.join(RESULTS_DIR, f"trivial_augment/{TASKS[task]}_results.csv")
    )
    results["task"] = task
    results["num_augmentations"] = results["num_augmentations"] - 1
    if i == 0:
        segmentation_results = results
    else:
        segmentation_results = pd.concat([segmentation_results, results])

    # Add the results for the best single augmentation and no augmentation
    individual_results = pd.read_csv(
        os.path.join(RESULTS_DIR, f"individual/{TASKS[task]}_results.csv")
    )
    best = individual_results.groupby("augmentation")["test/dice"].mean().idxmax()
    rows = individual_results[
        individual_results["augmentation"].isin([best, "identity"])
    ].copy()
    rows["task"] = task
    rows["num_augmentations"] = rows["augmentation"].apply(
        lambda x: 0 if x == "identity" else 1
    )
    rows.drop(columns=["augmentation"], inplace=True)
    segmentation_results = pd.concat([segmentation_results, rows])

# Calculate the mean and standard error of the mean for each metric (test/dice) for each task and number of augmentations
segmentation_results["num_augmentations"] = segmentation_results[
    "num_augmentations"
].astype("category")
segmentation_results["seed"] = segmentation_results["seed"].astype("category")
segmentation_results["task"] = segmentation_results["task"].astype("category")

# Group by task and number of augmentations and calculate the mean and standard error of the mean for each metric
segmentation_results = segmentation_results.drop(columns=["seed"])
segmentation_results = segmentation_results.groupby(["task", "num_augmentations"]).agg(
    {
        "test/loss": ["mean", "sem"],
        "test/dice": ["mean", "sem"],
    }
)
segmentation_results = segmentation_results.reset_index()
segmentation_results.columns = segmentation_results.columns = [
    "_".join(a).strip("_") for a in segmentation_results.columns.to_flat_index()
]
segmentation_results.round(3)

# %%
# Separate identity results from the rest of the results
identity_results = segmentation_results[segmentation_results["num_augmentations"] == 0]
identity_results = identity_results.drop(columns=["num_augmentations"])
identity_results.round(3)

# %%
# Calculate the difference between each number of augmentations and no augmentation
# (identity) for each metric (test/dice) for each task
results_df = segmentation_results.merge(
    identity_results, on="task", suffixes=("", "_identity")
)
results_df["test/loss_diff"] = (
    results_df["test/loss_mean"] - results_df["test/loss_mean_identity"]
)
results_df["test/dice_diff"] = (
    results_df["test/dice_mean"] - results_df["test/dice_mean_identity"]
)
results_df["test/dice_percent_change"] = (
    results_df["test/dice_diff"] / results_df["test/dice_mean_identity"] * 100
)
results_df.sort_index(axis=1).round(4)

# %%
summary_df = results_df[
    ["task", "num_augmentations"]
    + [col for col in results_df.columns if col.startswith("test/dice")]
]
summary_df = summary_df.drop(
    ["test/dice_mean_identity", "test/dice_sem_identity"], axis=1
)
summary_df = summary_df.rename(
    columns={
        "test/dice_mean": "Dice",
        "test/dice_sem": "Dice SE",
        "test/dice_diff": "Dice Diff",
        "test/dice_percent_change": "Dice % Change",
    }
)

# %%
task = "aul_liver_segmentation"
summary_df[summary_df["task"] == task].sort_values(
    "num_augmentations", ascending=True
).reset_index(drop=True).round(3)

# %%
summary_df[["num_augmentations", "Dice % Change"]].groupby(
    "num_augmentations"
).mean().round(2)

# %%
# For each task, plot the dice for each number of augmentations using a bar chart
titles = {
    "aul_liver_segmentation": "AUL Liver",
    "aul_mass_segmentation": "AUL Mass",
    "camus_segmentation": "CAMUS",
    "mmotu_segmentation": "MMOTU",
    "open_kidney_capsule_segmentation": "Open Kidney",
    "psfhs_segmentation": "PSFHS",
    "stanford_thyroid_segmentation": "Stanford Thyroid",
}

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, task in enumerate(TASKS):
    subset_df = summary_df[summary_df["task"] == task]

    identity_results = subset_df[subset_df["num_augmentations"] == 0]
    bottom = identity_results["Dice"].values[0]

    subset_df = subset_df.sort_values("num_augmentations")

    x_tick_labels = subset_df["num_augmentations"]
    x_ticks = np.arange(len(x_tick_labels))
    ax = axes[i // 4, i % 4]
    ax.scatter(
        x=x_ticks,
        y=subset_df["Dice"],
    )

    ax.errorbar(
        x=x_ticks,
        y=subset_df["Dice"],
        yerr=subset_df["Dice SE"],
        fmt="none",
        elinewidth=1,
        capsize=2,
    )

    ax.set_title(titles[task])
    ax.grid(axis="both", linestyle="--", linewidth=0.5, which="both")

    ax.set_xlabel("Number of Augmentations", fontsize=9)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        x_tick_labels,
        fontsize=8,
    )

    ax.set_ylabel("Dice", fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

axes[1, 3].axis("off")

plt.tight_layout()
plt.savefig("../outputs/figures/trivial_augment_segmentation.pdf")
plt.show()

# %%
