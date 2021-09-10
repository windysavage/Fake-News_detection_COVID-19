from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, hparams, train_ds, test_ds):
        self.batch_size = hparams["batch_size"]
        self.epochs = hparams["epochs"]
        self.lr = hparams["lr"]

        self.train_dataloader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_ds, batch_size=self.batch_size)
