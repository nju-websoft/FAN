from torch.utils.data import Dataset
import json
import logging
from loader.help import organize_bag, get_pos1_pos2
from transformers import BertTokenizer
import torch

logger = logging.getLogger('FAN')


class PU(Dataset):
    def __init__(self, config):
        super(PU, self).__init__()
        self.config = config
        self.max_len = config.max_len
        self.pos_data = self._read_file_(self.config.pos_file)
        self.unlabeled_data = self._read_file_(self.config.unlabeled_file)
        self.rel2id = json.load(open(self.config.rel2id, encoding='utf-8'))
        self.pos_bag = organize_bag(self.pos_data, max_bag_size=config.max_bag_size, use_label=True)
        self.unlabeled_bag = organize_bag(self.unlabeled_data, max_bag_size=config.max_bag_size, use_label=True)
        self.pos_size = len(self.pos_bag)  # positive samples
        self.unlabeled_size = len(self.unlabeled_bag)
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrain)

    def __len__(self):
        return 1000000000

    def __getitem__(self, item):
        """get pos-unlabeled pair"""
        pos_bag = self.pos_bag[item % self.pos_size]
        unlabeled_bag = self.unlabeled_bag[item % self.unlabeled_size]
        tuple1 = self._get_bag_(pos_bag)
        tuple2 = self._get_bag_(unlabeled_bag)
        return tuple1, tuple2

    def _get_bag_(self, bag):
        token2ids, pos1s, pos2s, masks, labels, confidence = [], [], [], [], [], []
        for record in bag:
            id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(record['text']))
            h_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(record['h']['name']))
            t_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(record['t']['name']))
            id = self.tokenizer.convert_tokens_to_ids(['[CLS]']) + id + self.tokenizer.convert_tokens_to_ids(['[SEP]'])
            pos1, pos2, h_pos, t_pos = get_pos1_pos2(id, h_id, t_id, self.config.pos_max)
            pre, post = min(h_pos, t_pos), max(h_pos, t_pos)

            mask = ([1] * pre + [2] * (post - pre) + [3] * (len(id) - post) + [0] * self.max_len)[:self.max_len]
            masks.append(mask)
            token2ids.append((id + [0] * self.max_len)[:self.max_len])
            pos1s.append((pos1 + [0] * self.max_len)[:self.max_len])
            pos2s.append((pos2 + [0] * self.max_len)[:self.max_len])
            labels.append(self.rel2id[record["relation"]])
            confidence.append(record["confidence"] if 'confidence' in record else 1)
        scope = len(token2ids)
        bag_labels = [0] * len(self.rel2id)
        for label in labels:
            bag_labels[label] = 1
        return torch.tensor(token2ids), torch.tensor(pos1s), torch.tensor(pos2s), torch.tensor(masks), \
               torch.tensor(labels), torch.tensor(confidence), torch.tensor(bag_labels), scope

    @staticmethod
    def _read_file_(file):
        array = []
        with open(file, encoding='utf-8') as f:
            for l in f:
                try:
                    record = json.loads(l)
                    array.append(record)
                except:
                    logging.debug('[DEBUG] loading json line')
                    pass
        return array

    def _generate_batch_(self, data):
        token2ids = torch.cat([t[0] for t in data], dim=0)
        pos1s = torch.cat([t[1] for t in data], dim=0)
        pos2s = torch.cat([t[2] for t in data], dim=0)
        masks = torch.cat([t[3] for t in data], dim=0)
        labels = torch.cat([t[4] for t in data], dim=0)
        confidence = torch.cat([t[5] for t in data], dim=0)
        bag_labels = torch.cat([t[6] for t in data], dim=0).reshape(-1, len(self.rel2id))
        begin = 0
        scope = []
        for s in [t[7] for t in data]:
            scope.append((begin, begin + s))
            begin += s
        return token2ids.to(self.config.device), pos1s.to(self.config.device), pos2s.to(self.config.device), \
               masks.to(self.config.device), labels.to(self.config.device), confidence.to(self.config.device), \
               bag_labels.to(self.config.device), scope

    def generate_batch(self, data):
        pos_batch = [t[0] for t in data]
        unlabeled_batch = [t[1] for t in data]
        tuple1 = self._generate_batch_(pos_batch)
        tuple2 = self._generate_batch_(unlabeled_batch)
        return tuple1, tuple2









