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

    def train(self):
        # 使用 Adam Optim 更新整個分類模型的參數
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_f = nn.BCELoss()

        THRESHOLD = 0.5
        epoch_loss = 0
        epoch_num = 0
        epoch_correct_num = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_labels = []
            epoch_preds = []

            for x, y in tqdm(iter(self.train_dataloader)):
                x = x["input_ids"].squeeze(1)

                x = x.to(self.device)
                y = y.to(self.device).to(dtype=torch.float)
                y = y.unsqueeze(1)

                # 將參數梯度歸零
                optimizer.zero_grad()

                # forward pass
                preds = self.model(x)
                loss = loss_f(preds, y)

                epoch_labels.extend(y.tolist())

                # backward
                loss.backward()
                optimizer.step()

                # 紀錄當前 batch loss
                epoch_loss += loss.item()
                epoch_num += x.size()[0]

                # 計算分類準確率
                preds = torch.gt(preds, THRESHOLD).to(dtype=torch.float)
                epoch_preds.extend(preds.tolist())
                epoch_correct_num += (preds == y).sum().item()

            epoch_f1 = f1_score(epoch_labels, epoch_preds)
            print(
                f"\nEpoch: {epoch+1}, Loss: {round(epoch_loss/epoch_num, 2)}, Acc: {round(epoch_correct_num/epoch_num, 2)}, F1_score: {round(epoch_f1, 2)}")
            # f1_test = test(model, test_dataloader)
