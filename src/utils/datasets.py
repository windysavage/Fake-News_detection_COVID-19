import torch
from torch.utils.data import Dataset


class SloganDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        self.xs = list(self.data["slogan"])
        self.xs = [tokenizer.encode_plus(
            x, max_length=64, truncation=True, padding="max_length", return_tensors="pt") for x in self.xs]

        self.ys = list(self.data.label)
        self.ys = [torch.tensor(y) for y in self.ys]

        self.examples = [(x, y) for x, y in zip(self.xs, self.ys)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
