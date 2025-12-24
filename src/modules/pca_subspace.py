import torch
import pytorch_lightning as pl

class PCASubspaceClassifier(pl.LightningModule):
    def __init__(self, k: int = 26, num_classes: int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.k = k
        self.num_classes = num_classes

        self.automatic_optimization = False

        self.register_buffer("class_means", torch.zeros(num_classes, 784))
        self.register_buffer("class_Uk", torch.zeros(num_classes, 784, k))
        self.fitted = False

    @torch.no_grad()
    def fit_closed_form(self, train_loader):
        X_list, y_list = [], []
        for xb, yb in train_loader:
            xb = xb.view(xb.size(0), -1).cpu()
            X_list.append(xb)
            y_list.append(yb.cpu())
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        for c in range(self.num_classes):
            Xc = X[y == c]
            mu = Xc.mean(dim=0)
            Z = Xc - mu
            cov = (Z.T @ Z) / (Z.size(0) - 1)

            evals, evecs = torch.linalg.eigh(cov)
            Uk = evecs[:, -self.k:]
            Uk = torch.flip(Uk, dims=[1])

            self.class_means[c] = mu # type: ignore
            self.class_Uk[c] = Uk # type: ignore

        self.fitted = True

    @torch.no_grad()
    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        Xc = x.unsqueeze(0) - self.class_means.unsqueeze(1) #type: ignore           # [C,B,784]
        n2 = (Xc * Xc).sum(dim=-1)                                     # [C,B]
        Y = torch.einsum("cbn,cnk->cbk", Xc, self.class_Uk)             # [C,B,k]
        y2 = (Y * Y).sum(dim=-1)                                       # [C,B]
        dist = n2 - y2

        return torch.argmin(dist, dim=0)

    def on_fit_start(self):
        if not self.fitted:
            self.fit_closed_form(self.trainer.datamodule.train_dataloader()) # type: ignore

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        acc = (pred == y).float().mean()
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        acc = (pred == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        acc = (pred == y).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return None
