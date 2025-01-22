"""Load the best model checkpoints for each seed, evaluate them on the test set, and report the results."""

import glob
import os

import hydra
import pandas as pd
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from usaugment.augmentation import get_data_loader_transforms
from usaugment.data import get_data_loader

SLURMEnvironment.detect = lambda: False


@hydra.main(version_base="1.3", config_path="config", config_name="test")
def main(config: DictConfig) -> None:
    # Check config for missing keys
    if OmegaConf.missing_keys(config):
        raise RuntimeError(
            f"Got missing keys in config:\n{OmegaConf.missing_keys(config)}"
        )

    # Configure transforms
    _, test_transform = get_data_loader_transforms(config)

    # Configure the data loader
    test_loader = get_data_loader(config, "test", test_transform, shuffle=False)

    # Evaluate each model checkpoint
    results = []
    checkpoint_paths = glob.glob(
        os.path.join(config.results_dir, "**/*.ckpt"), recursive=True
    )
    for checkpoint_path in checkpoint_paths:
        augmentation = checkpoint_path.removeprefix(config.results_dir).split("/")[0]
        seed = checkpoint_path.removeprefix(config.results_dir).split("/")[1]

        # Instantiate training module
        model = hydra.utils.get_class(config.model._target_).load_from_checkpoint(
            checkpoint_path, model=instantiate(config.model.model)
        )

        # Setup the trainer
        logger = CometLogger(
            project_name="usaugment-experiments",
            log_code=False,
            log_graph=False,
            auto_log_co2=False,
            log_env_details=False,
            offline=config.offline,
            save_dir=os.getcwd(),  # Hydra sets the working directory to the output directory,
            display_summary_level=0,
        )
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

        trainer = Trainer(
            max_epochs=config.epochs,
            deterministic=False,
            fast_dev_run=config.fast_dev_run,
            logger=logger,
            enable_progress_bar=config.enable_progress_bar,
            accumulate_grad_batches=config.accumulate_grad_batches,
            precision=config.precision,
        )

        # Train the model
        scores = trainer.test(model, test_loader, verbose=False)[0]
        scores["seed"] = int(seed)
        scores["augmentation"] = augmentation
        results.append(scores)

    # Create and display a DataFrame from the results
    df = pd.DataFrame(results)
    col = df.pop("augmentation")
    df.insert(0, col.name, col)
    col = df.pop("seed")
    df.insert(0, col.name, col)

    # Save the results to a CSV file
    df.to_csv(f"{config.task_name}_results.csv", index=False)

    # Display the mean and std. dev. of the key metric for each augmentation
    summary = df.groupby("augmentation")[f"test/{config.key_metric}"].agg(
        mean="mean", std="std"
    )
    summary["rank"] = summary["mean"].rank(ascending=False)
    summary = summary.sort_values(by="mean", ascending=False)
    summary = summary.reset_index()
    col = summary.pop("rank")
    summary.insert(0, col.name, col)
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
