import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lob.model import DeepLOB
from lob.data import DataModule
import torchmetrics

class DeepLOBModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DeepLOB()
        self.metrics = nn.ModuleDict({
            step: nn.ModuleDict({
                "acc": torchmetrics.Accuracy("multiclass", num_classes=3),
                "recall": torchmetrics.Recall("multiclass", num_classes=3, average="weighted"),
                "precision": torchmetrics.Precision("multiclass", num_classes=3, average="weighted"),
                "f1": torchmetrics.Accuracy("multiclass", num_classes=3, average="weighted"),
            }) for step in ["metric_train", "metric_valid", "metric_test"]
        })
        # self.metrics = {step: {
        #     "acc": torchmetrics.Accuracy("multiclass", num_classes=3),
        #     "recall": torchmetrics.Recall("multiclass", num_classes=3, average="weighted"),
        #     "precision": torchmetrics.Precision("multiclass", num_classes=3, average="weighted"),
        #     "f1": torchmetrics.Accuracy("multiclass", num_classes=3, average="weighted"),
        # } for step in ["train", "valid", "test"]}

    def _step(self, step, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log(f"{step}_loss", loss, prog_bar=True)
        for name, metric in self.metrics[f"metric_{step}"].items():
            metric(y_hat, y)
            self.log(f"{step}_{name}", metric, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("valid", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


if __name__ == "__main__":
    for k in range(5):
        for i in range(1, 10):
            base_name = f"model-{k}-{i}"
            checkpoint_callback = ModelCheckpoint(
                monitor="valid_loss",
                # dirpath="model_checkpoints/",
                filename=base_name+"-{epoch:02d}-{valid_loss:.2E}",
                save_top_k=3,  # -1 for saving all models
                mode="min",
            )
            early_stop_callback = EarlyStopping(
                monitor="valid_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
            
            trainer = pl.Trainer(
                max_epochs=50,
                callbacks=[checkpoint_callback, early_stop_callback],
            )
            model = DeepLOBModule()
            datamodule = DataModule(i=i, k=k)
            trainer.fit(model, datamodule)
            trainer.test(model, datamodule)
