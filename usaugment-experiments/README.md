# Revisiting Data Augmentation for Ultrasound Images

The code and documentation for reproducing the experiments presented in [Revisiting Data Augmentation for Ultrasound Images]().

## Requirements

To replicate our model training, install the required packages using the following command:

```bash
pip install -r cc_requirements.txt
```

To install the required packages for the evaluation, use the following command:

```bash
pip install -r requirements.txt
```

## Dataset Processing

The scripts required to process the datasets used in our evaluations are located in `scripts/data` alongside our code for the exploratory data analysis.

## Hyperparameter Tuning

The key regularization parameters (learning rate, weight decay, dropout rate, and number of epochs) were optimized for each task using Optuna. The Hyperparameter tuning for a particular task can be performed by running

```bash
sbatch tune.sh [DATSET_VERSION] [TASK_NAME]
```

which launches a 100-job job array on a Slurm cluster, where each job performs one trial. See `src/usaugment/tune.py` and `src/usaugment/config/` for more details.

## Individual Augmentation Evaluations

Evaluations of the individual data augmentations (30 repetitions of each of the 15 augmentations (incl. no augmentation) over the 14 tasks) can be performed by running:

```bash
bash launch_augmentation_evaluations.sh
```

This launches 14 job arrays (one for each task) containing 15 jobs (one for each augmentation). Each job sequentially performs 10 training runs, each with a different seed. See `src/usaugment/train.py` and `src/usaugment/config/` for more details.

The evaluations on the test set can then be performed using `src/usaugment/evaluate.py` 

## TrivialAugment Evaluations

The TrivialAugment evaluations (30 repetitions for each augmentation set over the 14 tasks) can be performed by running:

```bash
bash launch_trivial_augment_evaluations.sh
```

This launches 14 job arrays (one for each task) containing 13 jobs (one for each augmentation set). Each job sequentially performs 30 training runs, each with a different seed. See `src/usaugment/train_trivial_augment.py` and `src/usaugment/config/` for more details.

The evaluations on the test set can then be performed using `src/usaugment/evaluate_trivial_augment.py`

## Generating Figures and Results Tables

The code for generating each of the figures and tables in the paper are included in the `scripts/` directory.