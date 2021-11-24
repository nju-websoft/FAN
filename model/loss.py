import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_func(logits, labels)


class ConstrativeLoss(nn.Module):
    def __init__(self, tau):
        super(ConstrativeLoss, self).__init__()
        self.tau = tau

    def forward(self, bag_repre: torch.Tensor, rels: torch.Tensor):
        """
        :param bag_repre: (bags, hidden)
        :param rels: (bags, )
        :return:
        """
        h = bag_repre / torch.norm(bag_repre, dim=1).unsqueeze(-1)
        sim = torch.matmul(h, h.transpose(0, 1))
        idxs = (rels.unsqueeze(-1) == rels.unsqueeze(0)).float()

        pos_dist = self.tau - sim
        pos_dist *= (pos_dist > 0)
        pos_dist = pos_dist.square()
        pos, _ = torch.max((pos_dist - (1 - idxs) * 1e30), dim=1)
        neg_dist = sim * (sim > 0)
        neg_dist = neg_dist.square()
        neg, _ = torch.max(neg_dist - idxs * 1e30, dim=1)

        lctra = torch.sum(pos + neg, dim=0)
        return lctra


class BinaryLoss(nn.Module):
    def __init__(self, config):
        super(BinaryLoss, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.lf = nn.CrossEntropyLoss()

    def forward(self, logits):
        assert logits.size(0) == self.batch_size * 2
        binary_labels = torch.tensor([1] * self.batch_size + [0] * self.batch_size, dtype=torch.long)
        binary_labels = binary_labels.to(self.config.device)
        loss = self.lf(logits, binary_labels)
        return loss


