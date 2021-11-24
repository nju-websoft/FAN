from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import json
from filter_model import Filter
from data_loader import FilterLoader
import os


class Config:
    file = '../data/nyt10/train.txt'
    pretrain = '../bert/bert-base-uncased'
    epoch = 50
    theta = 0.5
    batch_size = 160
    dropout_rate = 0.5
    max_len = 120
    pos_max = 50
    pos_dim = 50
    filter_size = 230
    output_dir = '../output/mining/nyt10'
    tokenizer = 'bert'
    lr = 0.00001
    embedding_dim = 768
    stop_acc = 0.9
    checkpoint = '../output/mining/nyt10/MINING-0.9180.pkl'


def save_file(obj, path):
    dir, file = os.path.split(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path, 'w', encoding='utf-8') as f:
        for r in obj:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def predict():
    dataset = FilterLoader(Config)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=dataset._generate_batch_)

    model = Filter(Config).cuda()
    model.load_state_dict(torch.load(Config.checkpoint))
    model.train(False)

    pos, neg, unlabeled = [], [], []
    for i, data in tqdm(enumerate(data_loader), desc='[INFERRING]', total=len(dataset)):
        token2ids, masks, pos1s, pos2s, labels, records = data
        assert len(records) == 1
        if records[0]['relation'] != 'NA':
            pos.append(records[0])
            continue
        logits = model(token2ids, pos1s, pos2s, masks)
        if torch.sigmoid(logits) > Config.theta:
            unlabeled.append(records[0])
        else:
            neg.append(records[0])

    save_file(pos, os.path.join(Config.output_dir, 'pos.txt'))
    save_file(neg, os.path.join(Config.output_dir, 'neg.txt'))
    save_file(unlabeled, os.path.join(Config.output_dir, 'unlabeled.txt'))


if __name__ == '__main__':
    predict()




