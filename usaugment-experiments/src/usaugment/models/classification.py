import logging

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import (
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    Precision,
    PrecisionRecallCurve,
    Recall,
)
from torchvision.utils import make_grid

logger = logging.getLogger("lightning")


class ClassificationModel(L.LightningModule):
    def __init__(self, model, optimizer_partial, label_smoothing, task, num_classes=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimizer = optimizer_partial(params=model.parameters())
        self.label_smoothing = label_smoothing
        self.task = task
        self.num_classes = num_classes

        if task == "binary":
            self.loss_function = F.binary_cross_entropy_with_logits
        elif task == "multiclass":
            self.loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.train_precision = Precision(task=task, num_classes=num_classes)
        self.train_recall = Recall(task=task, num_classes=num_classes)
        self.train_f1 = F1Score(task=task, num_classes=num_classes)
        self.train_avg_precision = AveragePrecision(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_precision = Precision(task=task, num_classes=num_classes)
        self.val_recall = Recall(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task=task, num_classes=num_classes)
        self.val_avg_precision = AveragePrecision(task=task, num_classes=num_classes)
        self.val_pr_curve = PrecisionRecallCurve(task=task, num_classes=num_classes)
        self.val_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.test_acc = Accuracy(task=task, num_classes=num_classes)
        self.test_precision = Precision(task=task, num_classes=num_classes)
        self.test_recall = Recall(task=task, num_classes=num_classes)
        self.test_f1 = F1Score(task=task, num_classes=num_classes)
        self.test_avg_precision = AveragePrecision(task=task, num_classes=num_classes)
        self.test_pr_curve = PrecisionRecallCurve(task=task, num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)

        self.register_buffer("val_avg_precision_best", torch.tensor(0.0))

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self._log_images(x)

        y_hat = self(x).squeeze()
        loss = self.loss_function(y_hat, y.float() if self.task == "binary" else y)
        self.log("train/loss", loss)

        self.train_acc(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)
        self.train_avg_precision(y_hat, y)

        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=True, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=True, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train/avg_precision", self.train_avg_precision, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_function(y_hat, y.float() if self.task == "binary" else y)
        self.log("val/loss", loss)

        self.val_acc(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_avg_precision(y_hat, y)
        self.val_pr_curve(y_hat, y)
        self.val_confusion_matrix(y_hat, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_function(y_hat, y.float() if self.task == "binary" else y)
        self.log("test/loss", loss)

        self.test_acc(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_avg_precision(y_hat, y)
        self.test_pr_curve(y_hat, y)
        self.test_confusion_matrix(y_hat, y)

        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        val_avg_precision = self.val_avg_precision.compute()
        precision, recall, _ = self.val_pr_curve.compute()
        confusion_matrix = self.val_confusion_matrix.compute()

        # Log the scalar metrics
        self.log("val/acc", val_acc)
        self.log("val/precision", val_precision)
        self.log("val/recall", val_recall)
        self.log("val/f1", val_f1)
        self.log("val/avg_precision", val_avg_precision)

        if (not self.trainer.sanity_checking) and (val_avg_precision > self.val_avg_precision_best):
            # Log PR curve if the model is the best so far (this avoids Comet's hard limit on the number of curves that
            # can be logged)
            self.val_avg_precision_best = val_avg_precision
            if self.task == "binary":
                self.logger.experiment.log_curve(
                    name="val/pr_curve",
                    x=recall.cpu().tolist(),
                    y=precision.cpu().tolist(),
                    step=self.global_step,
                )
            else:
                for i in range(len(precision)):
                    self.logger.experiment.log_curve(
                        name=f"val/pr_curve_class{i}",
                        x=recall[i].cpu().tolist(),
                        y=precision[i].cpu().tolist(),
                        step=self.global_step,
                    )

        if not self.trainer.sanity_checking:
            # Log confusion matrix
            self.logger.experiment.log_confusion_matrix(
                name="val/confusion_matrix",
                matrix=confusion_matrix.cpu().tolist(),
                step=self.global_step,
            )

            # Log the current loss and average precision
            log_message = "Epoch {:3d}/{:d} | Avg. Precision = {:.3f} | Loss = {:3.3f}".format(
                self.current_epoch,
                self.trainer.max_epochs - 1,
                val_avg_precision,
                self.trainer.callback_metrics["val/loss"],
            )
            logger.info(log_message)

        # Reset the metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_avg_precision.reset()
        self.val_pr_curve.reset()
        self.val_confusion_matrix.reset()

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()
        test_avg_precision = self.test_avg_precision.compute()
        precision, recall, _ = self.test_pr_curve.compute()
        confusion_matrix = self.test_confusion_matrix.compute()

        # Log the scalar metrics
        self.log("test/acc", test_acc)
        self.log("test/precision", test_precision)
        self.log("test/recall", test_recall)
        self.log("test/f1", test_f1)
        self.log("test/avg_precision", test_avg_precision)

        # Log PR curve
        if self.task == "binary":
            self.logger.experiment.log_curve(
                name="test/pr_curve",
                x=recall.cpu().tolist(),
                y=precision.cpu().tolist(),
                step=self.global_step,
            )
        else:
            for i in range(len(precision)):
                self.logger.experiment.log_curve(
                    name=f"test/pr_curve_class{i}",
                    x=recall[i].cpu().tolist(),
                    y=precision[i].cpu().tolist(),
                    step=self.global_step,
                )

        # Log confusion matrix
        self.logger.experiment.log_confusion_matrix(
            name="test/confusion_matrix",
            matrix=confusion_matrix.cpu().tolist(),
            step=self.global_step,
        )

        # Reset the metrics
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_avg_precision.reset()
        self.test_pr_curve.reset()
        self.test_confusion_matrix.reset()

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


class BinaryClassificationModel(ClassificationModel):
    def __init__(self, model, optimizer_partial, label_smoothing):
        super().__init__(model, optimizer_partial, label_smoothing, task="binary")


class MultiClassClassificationModel(ClassificationModel):
    def __init__(self, model, optimizer_partial, num_classes, label_smoothing):
        super().__init__(model, optimizer_partial, label_smoothing, task="multiclass", num_classes=num_classes)
