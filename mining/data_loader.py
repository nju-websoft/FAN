import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer
import numpy as np
import logging
import os

logger = logging.getLogger('MINING')


class FilterLoader(Dataset):
    def __init__(self, config):
        super(FilterLoader, self).__init__()
        self.config = config
        self.max_len = config.max_len
        self.data = self._get_data_(config.file)
        if config.tokenizer == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config.pretrain)
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _get_data_(file):
        assert os.path.isfile(file)
        array = []
        with open(file, encoding='utf-8') as f:
            for l in f:
                try:
                    record = json.loads(l, encoding='utf-8')
                    array.append(record)
                except:
                    logger.debug('[DEBUG] loading json line')
                    pass
        return array

    def _generate_batch_(self, data):
        token2ids, masks, pos1s, pos2s, labels = [], [], [], [], []
        for record in data:
            id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(record['text']))
            h_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(record['h']['name']))
            t_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(record['t']['name']))
            pos1, pos2, h_pos, t_pos = get_pos1_pos2(id, h_id, t_id, self.config.pos_max)
            pre, post = min(h_pos, t_pos), max(h_pos, t_pos)
            mask = ([1] * pre + [2] * (post - pre) + [3] * (len(id) - post) + [0] * self.max_len)[:self.max_len]
            masks.append(mask)
            token2ids.append((id + [0] * self.max_len)[:self.max_len])
            pos1s.append((pos1 + [0] * self.max_len)[:self.max_len])
            pos2s.append((pos2 + [0] * self.max_len)[:self.max_len])
            labels.append(0 if record["relation"] == "NA" else 1)
        return torch.tensor(token2ids).cuda(), torch.tensor(masks).cuda(), torch.tensor(pos1s).long().cuda(), \
               torch.tensor(pos2s).long().cuda(), torch.tensor(labels).float().cuda(), data


def get_pos1_pos2(id, h_id, t_id, pos_max, closest=False):
    """
    >>> get_pos1_pos2([10, 5, 3, 1, 2, 10, 5, 9, 6, 8], [10, 5], [6, 8], 50)
    ([0, 1, 2, 2, 1, 0, 1, 2, 3, 4], [8, 7, 6, 5, 4, 3, 2, 1, 0, 1])
    """
    h_index, t_index = [], []
    len_h, len_t = len(h_id), len(t_id)
    for i in range(len(id)):
        if id[i: i + len_h] == h_id:
            h_index.append(i)
        if id[i: i + len_t] == t_id:
            t_index.append(i)
    assert len(h_index) > 0
    assert len(t_index) > 0
    h_pos, t_pos = h_index[0], t_index[0]

    h_index, t_index = np.array(h_index), np.array(t_index)
    pos1, pos2 = [0] * len(id), [0] * len(id)
    for i in range(len(id)):
        if closest:
            pos1[i] = min(min(abs(i - h_index) + 1), pos_max)
            pos2[i] = min(min(abs(i - t_index) + 1), pos_max)
        else:
            pos1[i] = min(abs(i - h_pos) + 1, pos_max)
            pos2[i] = min(abs(i - t_pos) + 1, pos_max)
    return pos1, pos2, h_pos, t_pos
