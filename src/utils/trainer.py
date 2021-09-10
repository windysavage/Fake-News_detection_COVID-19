import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer(object):
    def __init__(self, hparams, train_ds, test_ds, model, device):
        self.batch_size = hparams["batch_size"]
        self.epochs = hparams["epochs"]
        self.lr = hparams["lr"]

        self.train_dataloader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_ds, batch_size=self.batch_size)

        self.model = model
        self.device = device

        self.THRESHOLD = 0.5

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_f = nn.BCELoss()

        for epoch in range(self.epochs):
            self.model.train()

            epoch_loss = 0
            epoch_num = 0
            epoch_correct_num = 0
            epoch_labels = []
            epoch_preds = []

            for x, y in tqdm(iter(self.train_dataloader)):
                x = x["input_ids"].squeeze(1)

                x = x.to(self.device)
                y = y.to(self.device).to(dtype=torch.float)
                y = y.unsqueeze(1)

                optimizer.zero_grad()

                preds = self.model(x)
                loss = loss_f(preds, y)

                epoch_labels.extend(y.tolist())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_num += x.size()[0]

                preds = torch.gt(preds, self.THRESHOLD).to(dtype=torch.float)
                epoch_preds.extend(preds.tolist())
                epoch_correct_num += (preds == y).sum().item()

            epoch_f1 = f1_score(epoch_labels, epoch_preds)
            print(
                f"\nEpoch: {epoch+1}, Loss: {round(epoch_loss/epoch_num, 2)}, Acc: {round(epoch_correct_num/epoch_num, 2)}, F1_score: {round(epoch_f1, 2)}")

            self.test(epoch, loss_f)

    def test(self, epoch, loss_f):
        self.model.eval()

        epoch_loss = 0
        epoch_num = 0
        epoch_correct_num = 0

        epoch_labels = []
        epoch_preds = []

        with torch.no_grad():
            for x, y in tqdm(iter(self.test_dataloader)):
                x = x["input_ids"].squeeze(1)

                x = x.to(self.device)
                y = y.to(self.device).to(dtype=torch.float)
                y = y.unsqueeze(1)

                preds = self.model(x)
                loss = loss_f(preds, y)
                epoch_labels.extend(y.tolist())

                epoch_loss += loss.item()
                epoch_num += x.size()[0]

                preds = torch.gt(preds, self.THRESHOLD).to(dtype=torch.float)
                epoch_preds.extend(preds.tolist())
                epoch_correct_num += (preds == y).sum().item()

            epoch_f1 = f1_score(epoch_labels, epoch_preds)
            print(
                f"\nEpoch: {epoch+1}, Loss: {round(epoch_loss/epoch_num, 2)}, Acc: {round(epoch_correct_num/epoch_num, 2)}, F1_score: {round(epoch_f1, 2)}")
