import pytorch_lightning as pl

from data_module import MyDataModule
from transformer import Transformer

if __name__ == '__main__':
    dm = MyDataModule()
    model = Transformer()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)
