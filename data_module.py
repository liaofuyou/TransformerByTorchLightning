import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class Dimension:
    """
    各种大小
    """

    def __init__(self, batch=10, sents=32, token=30, vocab=2046) -> None:
        super().__init__()
        self.batch = batch
        self.sents = sents
        self.token = token
        self.vocab = vocab


class MyDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 10):
        super().__init__()

        source, target = self.get_data()

        self.train_set = TensorDataset(source, target)
        self.val_set = TensorDataset(source, target)
        self.test_set = TensorDataset(source, target)

        self.batch_size = batch_size

    @staticmethod
    def get_data(dim=Dimension()):
        data = np.random.randint(1, dim.vocab, size=(dim.batch * 10, dim.sents))
        data[:, 0] = 1
        return torch.LongTensor(data), torch.LongTensor(data)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
