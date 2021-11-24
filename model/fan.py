from torch import nn
import torch
from torch.autograd import Function
from transformers import BertModel
import json
from model.loss import CrossEntropyLoss
from model.loss import ConstrativeLoss
from model.loss import BinaryLoss


class ATT(nn.Module):
    def __init__(self, config, hidden, class_num):
        super(ATT, self).__init__()
        self.config = config
        self.linear = nn.Linear(hidden, class_num)

    def forward(self, x, labels, scopes):
        if self.training:
            query = torch.zeros_like(labels)
            for scope in scopes:
                query[scope[0]:scope[1]] = labels[scope[0]]
            relation_query = self.linear.weight[query]
            attention_logit = torch.sum(x * relation_query, 1, keepdim=True)
            tower_repre = []
            for start, end in scopes:
                sen_matrix = x[start: end]
                attention_score = torch.softmax(attention_logit[start: end].transpose(0, 1), dim=-1)
                final_repre = torch.matmul(attention_score, sen_matrix).squeeze()
                tower_repre.append(final_repre)
            stack_repre = torch.stack(tower_repre)
            logits = self.linear(stack_repre)
            return stack_repre, logits
        else:
            attention_logit = torch.matmul(x, self.linear.weight.transpose(0, 1))
            tower_repre = []
            for i, (start, end) in enumerate(scopes):
                sen_matrix = x[start: end]
                attention_score = torch.softmax(attention_logit[start: end].transpose(0, 1), dim=-1)
                final_repre = torch.matmul(attention_score, sen_matrix)
                logits = self.linear(final_repre)
                tower_repre.append(torch.diag(self.softmax(logits)))
            stack_repre = torch.stack(tower_repre)
            return stack_repre, stack_repre


class Discriminator(nn.Module):
    def __init__(self, config, in_features):
        super(Discriminator, self).__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, pos, unlabeled):
        x = torch.cat((pos, unlabeled), dim=0)
        x = self.layers(x)
        return x


class GRL(Function):
    @staticmethod
    def forward(ctx, x, ratio):
        ctx.ratio = ratio
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.ratio
        return output, None


class FAN(nn.Module):
    """False negative Adversarial Networks"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.rel2id = json.load(open(config.rel2id, encoding='utf-8'))
        self.ratio = -1.0
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.tau = config.tau

        self.bert = BertModel.from_pretrained(config.pretrain)
        self.pos_embedding = nn.Embedding(config.pos_max + 1, config.pos_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(config.word_dim + 2 * config.pos_dim, config.filter_size, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(config.word_dim + 2 * config.pos_dim, config.filter_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(config.word_dim + 2 * config.pos_dim, config.filter_size, kernel_size=4, padding=2)
        self.conv4 = nn.Conv1d(config.word_dim + 2 * config.pos_dim, config.filter_size, kernel_size=5, padding=2)
        self.mask_embedding = nn.Embedding.from_pretrained(
            torch.tensor([[-100, -100, -100], [0, -100, -100], [-100, 0, -100], [-100, -100, 0]], dtype=torch.float),
            freeze=True)

        self.pool = nn.MaxPool2d((config.max_len, 1))
        self.non_linear = nn.ReLU()
        self.dropout = nn.Dropout()
        self.aggregation = ATT(config, 4 * 3 * config.filter_size, len(self.rel2id))
        self.discriminator = Discriminator(config, 4 * 3 * config.filter_size)

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

    def forward(self, pos, unlabeled):
        """
        :param pos: token2ids, pos1s, pos2s, masks, labels, confidence, bag_labels, scope
        :param unlabeled:
        :return:
        """
        pt, pp1, pp2, pm, pl, pc, pbl, ps = pos
        ut, up1, up2, um, ul, uc, ubl, us = unlabeled

        token2ids = torch.cat((pt, ut), dim=0)
        pos1s = torch.cat((pp1, up1), dim=0)
        pos2s = torch.cat((pp2, up2), dim=0)
        masks = torch.cat((pm, um), dim=0)
        labels = torch.cat((pl, ul), dim=0)
        scopes = self.merge_scope(ps, us)

        if self.config.freeze:
            with torch.no_grad():
                x = self.bert(token2ids, masks)
        else:
            x = self.bert(token2ids, masks)
        x = x.last_hidden_state
        x1 = self.pos_embedding(pos1s)
        x2 = self.pos_embedding(pos2s)
        x = torch.cat((x, x1, x2), dim=-1)
        x = self.convolution(x.transpose(1, 2))
        x = self.piecewise_max_pooling(x, masks)
        # ===== Logits_Classification =====
        feature, logits = self.aggregation(x, labels, scopes)
        p, u = feature[:self.batch_size], feature[self.batch_size:]
        logits_p, logits_u = logits[:self.batch_size], logits[self.batch_size:]
        assert p.shape == u.shape
        u = GRL.apply(u, self.ratio)
        # ===== Logits_Discriminator =====
        d_logits = self.discriminator(p, u)

        return {
            'feature': feature,
            'p': p,
            'u': u,
            'logits': logits,
            'logits_p': logits_p,
            'logits_u': logits_u,
            'd_logits': d_logits,
            'bag_labels_p': pbl,
            'bag_labels_u': ubl
        }

    def compute_loss(self, output_dict):
        cls_lf = CrossEntropyLoss(self.config)
        ctra_lf = ConstrativeLoss(self.tau)
        d_lf = BinaryLoss(self.config)

        # ===== Loss_Classification =====
        pbl = output_dict.get('bag_labels_p', None)
        flare = []
        for bl in pbl:
            flare.append(torch.nonzero(bl == 1)[0])
        flare = torch.cat(flare, dim=0)
        logits_p = output_dict.get('logits_p', None)
        l1 = cls_lf(logits_p, flare)
        # ===== Loss_Discriminator =====
        d_logits = output_dict.get('d_logits', None)
        l2 = d_lf(d_logits)
        # ===== Loss_Contrastive =====
        p = output_dict.get('p', None)
        l3 = ctra_lf(p, flare)
        # ===== Loss_Summary =====
        l = l1 + self.beta * l2 + self.gamma * l3
        return l, l1, l2, l3

    @staticmethod
    def merge_scope(scope1, scope2):
        """
        :param scope1: [(0, 3), (3, 5)]
        :param scope2: [(0, 1), (1, 2)]
        :return: [(0, 3), (3, 5), (5, 6), (6, 7)]
        """

        pos_end = scope1[-1][1]
        tail = [(t[0] + pos_end, t[1] + pos_end) for t in scope2]
        return scope1 + tail
