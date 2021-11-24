import os
import logging
import socket
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
import argparse
import random
import sys
import time
from model.fan import FAN
from loader.pu import PU
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logging(args):
    formatter = logging.Formatter(
        '[%(asctime)-15s] %(levelname)s-%(filename)s-line %(lineno)d: %(message)s',
        '%d %b %Y %H:%M:%S')
    logger = logging.getLogger('FAN')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    log_path = os.path.join(args.output_dir, 'FAN-{}.log'.format(time.strftime('%y-%m-%d_%H_%M', time.localtime())))
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main(args):
    set_seed(args.seed)
    logger = set_logging(args)

    logger.info('============== Configurations ==============')
    for key, val in vars(args).items():
        logger.info('{}: {}'.format(key, val))
    logger.info('============================================')

    dataset = PU(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.generate_batch)
    model = FAN(args)
    model = model.to(args.device)

    logger.info('============== Parameters ==============')
    for k, v in model.named_parameters():
        logger.info(k)
    logger.info('========================================')

    params1 = [v for k, v in model.named_parameters() if k.startswith('bert')]
    params2 = [v for k, v in model.named_parameters() if not k.startswith('bert')]
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([
            {"params": params1, "lr": 1e-7},
            {"params": params2}],
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    total_steps = args.epoch * dataset.pos_size // args.batch_size
    global_step = 0
    for _, (pos, unlabeled) in enumerate(data_loader):
        model.train()
        output_dict = model(pos, unlabeled)
        loss, l1, l2, l3 = model.compute_loss(output_dict)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step >= total_steps:
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'aligning/fan.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='output dir for results, log, checkpoint, et al.')
    parser.add_argument('--device', type=str)
    parser.add_argument('--pos_file', type=str, help='positive samples')
    parser.add_argument('--unlabeled_file', type=str, help='unlabeled samples in N/A')
    parser.add_argument('--rel2id', type=str, help='relation to id file')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_len', type=int, default=120, help='max length of sentences')
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 penalization')
    parser.add_argument('--word_dim', type=int, default=768)
    parser.add_argument('--pos_max', type=int, default=50)
    parser.add_argument('--pos_dim', type=int, default=50)
    parser.add_argument('--filter_size', default=230, type=int)
    parser.add_argument('--contrastive', action='store_true', help='contrastive learning')
    parser.add_argument('--alpha', type=float, default=0.0001, help='balance efficient for GRL')
    parser.add_argument('--beta', type=float, default=0.1, help='balance efficient for discriminator')
    parser.add_argument('--gamma', type=float, default=0.0001, help='coefficient for contrastive learning')
    parser.add_argument('--tau', type=float, default=1.0, help='avoid representation collapse')
    parser.add_argument('--weighting', default=False, action='store_true', help='add weight in classification')
    parser.add_argument('--max_bag_size', default=-1, type=int, help='max bag size, -1 means keep original size')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    main(args)






