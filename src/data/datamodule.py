import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision import datasets
import torchvision.transforms.v2 as T


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        for_transformer: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.for_transformer = for_transformer

        if for_transformer:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        full_train = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=self.transform,
        )

        self.train_set, self.val_set = random_split(full_train, [55000, 5000])

        self.test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
