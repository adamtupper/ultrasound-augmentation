# Revisiting Data Augmentation for Ultrasound Images

This repository contains the code and documentation for reproducing the experiments presented in [Revisiting Data Augmentation for Ultrasound Images]().

> [!IMPORTANT]
> [UltraBench](https://github.com/adamtupper/ultrabench) and [USAugment](https://github.com/adamtupper/usaugment) have been released as standalone Python packages so that they can be used, updated and maintained more easily. The original versions are retained in this repository for the sole purpose of reproducing our original experiments. If you want to use UltraBench or USAugment we recommend using the maintained packaged versions.

## Getting Started

This repository uses the Visual Studio Code [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers) for containerized development (the configuration is in the `.devcontainer` directory). To replicate our environment, run the **Dev Containers: Reopen in Container** command after opening the project in VS Code (you may need to install the extension first).

All model training was performed on a Slurm cluster using an NVIDIA V100 GPU, 24 to 32 GB of RAM, and 15 CPU cores. The scripts for repeating these experiments on such a cluster are in the `scripts/slurm` directory.

### Requirements

The requirements for preprocessing the datasets and processing the results are listed in the `requirements.txt` file. Furthermore, the requirements for training the models are listed in the `requirements_cc.txt` file. We recommend installing these in an isolated virtual environment.

## Replicating the Experiments

### 1. Dataset Setup

The exploratory data analysis (EDA) and dataset preprocessing scripts for each of the 10 datasets are in the `scripts/data` directory. The EDA scripts include download links for the datasets and accompanying articles (where applicable).

The `prepare_[DATASET].py` script must be run before running experiments on a dataset. The docstring at the start of the file describes how to run each script.

### 2. Hyperparameter Tuning

The key regularization parameters (learning rate, weight decay, dropout rate, and number of epochs) were optimized for each task using Optuna. A single trial is performed using `src/usaugment/tune.py`. For example, the hyperparameters for the AUL liver segmentation task are tuned by running:

```bash
python src/usaugment/tune.py \
    output_dir=path/to/save/outputs \
    data_dir=path/to/preprocessed/dataset \
    task=aul_liver_segmentation
```

The list of task names is located in the `src/usaugment/config/task` directory. The script is configured using [Hydra](https://hydra.cc/). The configuration used in our experiments is the default and is read from the `src/usaugment/config` directory. You can view all the configuration options by using the `--help` flag.

See `scripts/slurm/tune.sh` for running all trials on a Slurm cluster.

### 3. Evaluating Individual Augmentations

#### Model Training

The `/src/usaugment/train.py` script trains a model for a specific task using a single augmentation. For example, to train a model using rotate on the AUL liver segmentation task use:

```bash
python src/usaugment/train.py \
        output_dir=path/to/save/outputs \
        data_dir=path/to/preprocessed/dataset \
        task=aul_liver_segmentation \
        augmentation=rotate \
        seed=0
```

The list of augmentations is located in the `src/usaugment/config/augmentation` directory. The script is configured using [Hydra](https://hydra.cc/). The configuration used in our experiments is the default and is read from the `src/usaugment/config` directory. You can view all the configuration options by using the `--help` flag.

See `/scripts/slurm/launch_augmentation_evaluations.sh` for an example of how to train the models over all tasks, augmentations, and seeds.

#### Model Evaluation

After training, you can can evaluate a model on the test split using the `src/usaugment/evaluate.py` script. This script evaluates each model trained using each augmentation and generates a CSV file with the results. For example,

```bash
python src/usaugment/evaluate.py \
    data_dir=path/to/preprocessed/dataset \
    output_dir=path/to/save/outputs \
    task=aul_liver_segmentation \
    +results_dir=path/to/training/results/dir
```

### 4. Evaluating TrivialAugment

#### Model Training

The `src/usaugment/train_trivial_augment.py` script trains a model for a specific task using a specific TrivialAugment configuration. For example, to train a model using TrivialAugment with the Top-3 augmentations on the AUL liver segmentation task use:

```bash
python src/usaugment/train.py \
        output_dir=path/to/save/outputs \
        data_dir=path/to/prepocessed/dataset \
        task=aul_liver_segmentation \
        augmentation=trivial_augment_aul_liver_segmentation \
        seed=0 \
        top_n_augmentations=4  # Use N + 1 since the first augmentation is always Identity
```

The script is configured using [Hydra](https://hydra.cc/). The configuration used in our experiments is the default and is read from the `src/usaugment/config` directory. You can view all the configuration options by using the `--help` flag.

See `/scripts/slurm/launch_trivial_augment_evaluations.sh` for an example of how to train models over all tasks, augmentation sets, and seeds.

#### Model Evaluation

After training, you can can evaluate a model on the test split using the `src/usaugment/evaluate_trivial_augment.py` script. This script evaluates each model trained using each augmentation set and generates a CSV file with the results. For example,

```bash
python src/usaugment/evaluate_trivial_augment.py \
    data_dir=path/to/preprocessed/dataset \
    output_dir=path/to/save/outputs \
    task=aul_liver_segmentation \
    +results_dir=path/to/training/results/dir
```

### 5. Generating the Figures and Results Tables

The code for generating each of the figures and tables in the paper are included in the `scripts/results` directory.

## Citing Our Work

If you use our code for your research, please cite our paper!

```
```

## Questions & Contributions

We welcome contributions to the [UltraBench](https://github.com/adamtupper/ultrabench) and [USAugment](https://github.com/adamtupper/usaugment) packages released with this work. For more information, please visit the dedicated repositories.

If you have any questions related to trying to replicate our experiments, please [open an issue](https://github.com/adamtupper/usaugment-experiments/issues) or email us ([adam.tupper.1@ulaval.ca](mailto:adam.tupper.1@ulaval.ca)) and we'll do our best to help you.