from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import logging
import sys
import time
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


def set_logging():
    formatter = logging.Formatter(
        '[%(asctime)-15s] %(levelname)s-%(filename)s-line %(lineno)d: %(message)s',
        '%d %b %Y %H:%M:%S')
    logger = logging.getLogger('MINING')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    log_path = 'MINING-{}.log'.format(time.strftime('%y-%m-%d_%H_%M', time.localtime()))
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train():
    dataset = FilterLoader(Config)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=Config.batch_size, collate_fn=dataset._generate_batch_)

    model = Filter(Config).cuda()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.00001)

    logger = logging.getLogger('MINING')
    for epoch in range(Config.epoch):
        label_history, pred_history = [], []
        model.train()
        for i, data in tqdm(enumerate(data_loader), desc='[TRAINING]', total=len(dataset) // Config.batch_size):
            token2ids, masks, pos1s, pos2s, labels, _ = data
            logits = model(token2ids, pos1s, pos2s, masks)
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            label_history += labels.cpu().tolist()
            pred_history += (torch.sigmoid(logits) > 0.5).long().cpu().tolist()

        acc = accuracy_score(label_history, pred_history)
        macro_f1 = f1_score(label_history, pred_history, average='macro')
        micro_f1 = f1_score(label_history, pred_history, average='micro')
        logger.info("epoch {}, accuracy {:.4f}, macro-f1 {:.4f}, micro-f1 {:.4f}".format(epoch, acc, macro_f1, micro_f1))
        if acc > Config.stop_acc:
            if not os.path.exists(Config.output_dir):
                os.makedirs(Config.output_dir)
            torch.save(model.state_dict(), os.path.join(Config.output_dir, "MINING-{:.4f}.pkl".format(acc)))
            break


if __name__ == '__main__':
    set_logging()
    train()

