import os

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from usaugment.augmentation import get_test_transform, get_trivial_augment_transform
from usaugment.data import get_data_loader

SLURMEnvironment.detect = lambda: False


@hydra.main(version_base="1.3", config_path="config", config_name="train_trivial_augment")
def main(config: DictConfig) -> None:
    # Check config for missing keys
    if OmegaConf.missing_keys(config):
        raise RuntimeError(f"Got missing keys in config:\n{OmegaConf.missing_keys(config)}")

    # Seed everything
    L.seed_everything(config.seed)

    # Configure transforms
    train_transform, test_transform = get_trivial_augment_transform(config), get_test_transform(config)

    # Configure data loaders
    train_loader = get_data_loader(config, "train", train_transform, shuffle=True)
    val_loader = get_data_loader(config, "validation", test_transform, shuffle=False)

    # Instantiate training module
    model = instantiate(config.model)

    # Setup the trainer
    checkpoint_callback = ModelCheckpoint(monitor=f"val/{config.key_metric}", mode="max", save_top_k=1)
    logger = CometLogger(
        project_name="ultrasound-augmentation",
        log_code=False,
        log_graph=False,
        auto_log_co2=False,
        log_env_details=False,
        offline=config.offline,
        save_dir=os.getcwd(),  # Hydra sets the working directory to the output directory
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    trainer = Trainer(
        max_epochs=config.epochs,
        deterministic=False,
        fast_dev_run=config.fast_dev_run,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=logger,
        enable_progress_bar=config.enable_progress_bar,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
        log_every_n_steps=len(train_loader),
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
