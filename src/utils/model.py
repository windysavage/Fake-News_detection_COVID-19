import torch.nn as nn


class FinetuneBert(nn.module):
    def __init__(self, bert):
        super(FinetuneBert, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=0.2)

        self.out = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bert(x)
        x = self.dropout(x.pooler_output)
        x = self.out(x)
        x = self.sigmoid(x)

        return x
