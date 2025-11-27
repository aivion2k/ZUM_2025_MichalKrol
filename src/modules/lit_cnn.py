from typing import Any
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.cnn import MNISTCNN


class LitMNISTCNN(LightningModule):
    """PyTorch Lightning module for MNIST classification using a CNN."""

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = MNISTCNN(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr) # type: ignore