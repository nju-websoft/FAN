from collections import defaultdict
import numpy as np


def organize_bag(data, max_bag_size, use_label):
    key2records = defaultdict(list)
    for record in data:
        if use_label:
            key = (record['h']['name'], record['t']['name'], record['relation'])
        else:
            key = (record['h']['name'], record['t']['name'])
        key2records[key].append(record)
    bag_list = []
    for bag in key2records.values():
        if 0 < max_bag_size < len(bag):
            bag_list.append(np.random.choice(bag, size=max_bag_size, replace=False))
        else:
            bag_list.append(bag)
    return bag_list


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








