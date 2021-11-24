import torch
from torch import nn
from transformers import BertModel


class Filter(nn.Module):
    def __init__(self, config):
        super(Filter, self).__init__()
        self.config = config
        if config.tokenizer == 'bert':
            self.word_embedding = BertModel.from_pretrained(config.pretrain)
        else:
            raise NotImplementedError

        self.pos_embedding = nn.Embedding(config.pos_max + 1, config.pos_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(config.embedding_dim + 2 * config.pos_dim, config.filter_size, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(config.embedding_dim + 2 * config.pos_dim, config.filter_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(config.embedding_dim + 2 * config.pos_dim, config.filter_size, kernel_size=4, padding=2)
        self.conv4 = nn.Conv1d(config.embedding_dim + 2 * config.pos_dim, config.filter_size, kernel_size=5, padding=2)

        self.mask_embedding = nn.Embedding.from_pretrained(
            torch.tensor([[-100, -100, -100], [0, -100, -100], [-100, 0, -100], [-100, -100, 0]], dtype=torch.float),
            freeze=True)

        self.pool = nn.MaxPool2d((config.max_len, 1))
        self.non_linear = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(4 * 3 * config.filter_size, 100)
        self.linear2 = nn.Linear(100, 1)

        self.init()

    def init(self):
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)

    def convolution(self, x):
        seq_len = x.size(2)
        t1 = self.conv1(x)[:, :, :seq_len]
        t2 = self.conv2(x)
        t3 = self.conv3(x)[:, :, :seq_len]
        t4 = self.conv4(x)
        x = torch.cat((t1, t2, t3, t4), dim=1)
        return x

    def piecewise_max_pooling(self, x, masks):
        x = x.transpose(1, 2).unsqueeze(-1)
        masks = self.mask_embedding(masks).unsqueeze(-2)
        x = self.non_linear((x + masks).transpose(1, 2))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, token2ids, pos1s, pos2s, masks):
        x = self.word_embedding(token2ids, masks)[0]  # (batch, seq_len, hidden)
        x = self.dropout(x)
        p1 = self.pos_embedding(pos1s)
        p2 = self.pos_embedding(pos2s)
        x = torch.cat((x, p1, p2), dim=2)
        x = self.convolution(x.transpose(1, 2))
        x = self.piecewise_max_pooling(x, masks)  # (batch, 4 * 3 * filter_size)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.non_linear(x)
        x = self.linear2(x).squeeze()  # (batch, )
        return x







