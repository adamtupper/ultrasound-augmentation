import logging

import lightning as L
import torch
from monai.losses import DiceLoss
from monai.networks.utils import one_hot
from torch import optim
from torchmetrics import Dice
from torchvision.utils import make_grid

logger = logging.getLogger("lightning")


class SegmentationModel(L.LightningModule):
    def __init__(self, model, optimizer_partial):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimizer = optimizer_partial(params=model.parameters())

        self.loss_function = DiceLoss(sigmoid=True, include_background=False)
        self.train_dice = Dice(ignore_index=0)
        self.val_dice = Dice(ignore_index=0)
        self.test_dice = Dice(ignore_index=0)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            log_message = "Epoch {:3d}/{:d} | Dice = {:.3f} | Loss = {:3.3f}".format(
                self.current_epoch,
                self.trainer.max_epochs - 1,
                self.trainer.callback_metrics["val/dice"],
                self.trainer.callback_metrics["val/loss"],
            )
            logger.info(log_message)

    def configure_optimizers(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.trainer.estimated_stepping_batches)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def _log_images(self, x):
        image_grid = make_grid(x, nrow=8).permute(1, 2, 0).cpu()
        self.logger.experiment.log_image(name="train/x", image_data=image_grid, step=self.global_step)


class BinarySegmentationModel(SegmentationModel):
    def __init__(self, model, optimizer_partial):
        super().__init__(model, optimizer_partial)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self._log_images(x)

        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss)

        self.train_dice(y_hat, y)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if batch_idx == 0:
            self._log_predictions(x, y_hat, y)

        loss = self.loss_function(y_hat, y)
        self.log("val/loss", loss)

        self.val_dice(y_hat, y)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("test/loss", loss)

        self.test_dice(y_hat, y)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True)

        return loss

    def _log_predictions(self, x, y_hat, y):
        image_grid = make_grid(x, nrow=8).permute(1, 2, 0).cpu()

        preds = (y_hat > 0.5).float()

        true_masks = y.repeat(1, 3, 1, 1)
        true_masks[:, 0] = 0.0
        true_masks[:, 2] = 0.0

        predicted_masks = preds.repeat(1, 3, 1, 1)
        predicted_masks[:, 1] = 0.0
        predicted_masks[:, 2] = 0.0

        overlay = true_masks + predicted_masks
        overlay_grid = make_grid(overlay, nrow=8).permute(1, 2, 0).cpu()

        self.logger.experiment.log_image(name="val/masks", image_data=overlay_grid, step=self.global_step)
        self.logger.experiment.log_image(name="val/x", image_data=image_grid, step=self.global_step)


class MultiClassSegmentationModel(SegmentationModel):
    def __init__(self, model, optimizer_partial, num_classes):
        super().__init__(model, optimizer_partial)
        self.num_classes = num_classes

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self._log_images(x)

        y_hat = self(x)
        y = one_hot(y, num_classes=self.num_classes, dtype=torch.int)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss)

        self.train_dice(y_hat, y)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = one_hot(y, num_classes=self.num_classes, dtype=torch.int)

        if batch_idx == 0:
            self._log_predictions(x, y_hat, y)

        loss = self.loss_function(y_hat, y)
        self.log("val/loss", loss)

        self.val_dice(y_hat, y)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = one_hot(y, num_classes=self.num_classes, dtype=torch.int)
        loss = self.loss_function(y_hat, y)
        self.log("test/loss", loss)

        self.test_dice(y_hat, y)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True)

        return loss

    def _log_predictions(self, x, y_hat, y):
        image_grid = make_grid(x, nrow=8).permute(1, 2, 0).cpu()

        for i in range(self.num_classes):
            preds = y_hat.argmax(dim=1).float()

            true_masks = y[:, i].unsqueeze(1).repeat(1, 3, 1, 1)
            true_masks[:, 0] = 0.0
            true_masks[:, 2] = 0.0

            predicted_masks = (preds == i).unsqueeze(1).repeat(1, 3, 1, 1)
            predicted_masks[:, 1] = 0.0
            predicted_masks[:, 2] = 0.0

            overlay = true_masks + predicted_masks
            overlay_grid = make_grid(overlay, nrow=8).permute(1, 2, 0).cpu()

            self.logger.experiment.log_image(name=f"val/masks_class{i}", image_data=overlay_grid, step=self.global_step)

        self.logger.experiment.log_image(name="val/x", image_data=image_grid, step=self.global_step)
