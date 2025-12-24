import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchmetrics.classification import MulticlassAccuracy

class ViTLightning(pl.LightningModule):
    def __init__(self, model_name="google/vit-base-patch16-224", lr=2e-5, wd=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=10, ignore_mismatched_sizes=True
        )
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.val_acc = MulticlassAccuracy(num_classes=10)
        self.test_acc = MulticlassAccuracy(num_classes=10)

    def _to_vit_pixel_values(self, x):
        # x: [B,1,224,224] float in [0,1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # -> [B,3,224,224]
        # processor expects images either PIL or torch tensors.
        # For torch tensors: use do_rescale=False because already [0,1]
        enc = self.processor(images=x, return_tensors="pt", do_rescale=False)
        return enc["pixel_values"].to(x.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pixel_values = self._to_vit_pixel_values(x)
        out = self.model(pixel_values=pixel_values, labels=y)
        preds = out.logits.argmax(dim=-1)
        self.log("train_loss", out.loss, prog_bar=True)
        self.log("train_acc", self.train_acc(preds, y), prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pixel_values = self._to_vit_pixel_values(x)
        out = self.model(pixel_values=pixel_values, labels=y)
        preds = out.logits.argmax(dim=-1)
        self.log("val_loss", out.loss, prog_bar=True)
        self.log("val_acc", self.val_acc(preds, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pixel_values = self._to_vit_pixel_values(x)
        out = self.model(pixel_values=pixel_values, labels=y)
        preds = out.logits.argmax(dim=-1)
        self.log("test_acc", self.test_acc(preds, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd) # type: ignore


early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=1,          # 1–2 na MNIST zwykle wystarczy
    min_delta=1e-4
)