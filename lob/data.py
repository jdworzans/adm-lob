import numpy as np
from lightning.pytorch.core import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils import data

DATA_DIR = "data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre"
TRAIN_DIR = f"{DATA_DIR}/NoAuction_DecPre_Training"
TEST_DIR = f"{DATA_DIR}/NoAuction_DecPre_Testing"

def load(filepath: str):
    data = np.loadtxt(filepath)
    X = torch.as_tensor(data[:40, :].T).float()
    y = torch.as_tensor(data[-5:, :].T)
    return X, y

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T

        self.X = torch.as_tensor(data[:40, :].T).float()
        self.y = torch.as_tensor(data[-5:, :].T)[:, self.k].long() - 1

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X) - self.T + 1

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.X[index:index + self.T, :].unsqueeze(0), self.y[index + self.T - 1]

class DataModule(LightningDataModule):
    def __init__(
        self, i: int = 1, k: int = 4, num_classes: int = 3,
        T: int = 100, batch_size: int = 64
    ):
        super().__init__()
        train_data = np.loadtxt(f'{TRAIN_DIR}/Train_Dst_NoAuction_DecPre_CF_{i}.txt')
        n_train_samples = int(np.floor(train_data.shape[1] * 0.8))
        val_data = train_data[:, n_train_samples:]
        train_data = train_data[:, :n_train_samples]
        test_data = np.loadtxt(f'{TEST_DIR}/Test_Dst_NoAuction_DecPre_CF_{i}.txt')
        self.train_dataset = Dataset(train_data, k, num_classes, T)
        self.val_dataset = Dataset(val_data, k, num_classes, T)
        self.test_dataset = Dataset(test_data, k, num_classes, T)

        self.batch_size = batch_size
        self.num_workers = 8

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
