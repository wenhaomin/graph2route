# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import math

"""
Evaluation Function
"""


# For eta task
def eta_eval(pred, label, metric='mae'):
    mask = label > 0
    label = label.masked_select(mask)
    pred = pred.masked_select(mask)
    if metric == 'mae': result = nn.L1Loss()(pred, label).item()
    if metric == 'mse': result = nn.MSELoss()(pred, label).item()
    if metric == 'rmse':
        mse = nn.MSELoss()(pred, label).item()
        result = np.sqrt(mse)
    if metric == 'mape': result = torch.abs((pred - label) / label).mean().item()
    if 'acc' in metric:  # calculate the Hit@10min, Hit@20min
        k = int(metric.split('@')[1])
        tmp = torch.abs(pred - label) < k
        result = torch.sum(tmp).item() / tmp.shape[0]
        #result = result * 100
    n = mask.sum().item()
    return result, n


# For sorting task
def hit_rate(pred, label, lab_len, top_n=3):
    """
    Get the top-n hit rate of the prediction
    :param lab_len:
    :param pred:
    :param label:
    :param top_n:
    :return:
    """
    # label_len = get_label_len(label)
    label_len = lab_len
    eval_num = min(top_n, label_len)
    hit_num = len(set(pred[:eval_num]) & set(label[:eval_num]))
    hit_rate = hit_num / eval_num
    return hit_rate


def kendall_rank_correlation(pred, label, label_len):
    """
    caculate the kendall rank correlation between pred and label, note that label set is contained in the pred set
    :param label_len:
    :param pred:
    :param label:
    :return:
    """
    # print('pred:', pred)
    # print('label:', label)
    # print('label len:', label_len)


    def is_concordant(i, j):
        return 1 if (label_order[i] < label_order[j] and pred_order[i] < pred_order[j]) or (
                label_order[i] > label_order[j] and pred_order[i] > pred_order[j]) else 0

    if label_len == 1: return 1

    label = label[:label_len]
    not_in_label = set(pred) - set(label)# 0
    # get order dict
    pred_order = {d: idx for idx, d in enumerate(pred)}
    label_order = {d: idx for idx, d in enumerate(label)}
    for o in not_in_label:
        label_order[o] = len(label)
    # print('label order:', label_order)

    n = len(label)
    # compare list 1: compare items between labels
    lst1 = [(label[i], label[j]) for i in range(n) for j in range(i + 1, n)]
    # compare list 2: compare items between label and pred
    lst2 = [(i, j) for i in label for j in not_in_label]

    try:
        hit_lst = [is_concordant(i, j) for i, j in (lst1 + lst2)]
    except:
        print('pred:', pred)
        print('label:', label)
        print('label len:', label_len)
        print('-' * 40)
        return float(1)
        # hit_list = [0 for i, j in (lst1 + lst2)]


    # hit_lst = [is_concordant(i, j) for i, j in (lst1 + lst2)]
    # todo_: add the weight here
    hit = sum(hit_lst)
    not_hit = len(hit_lst) - hit
    result = (hit - not_hit) / (len(lst1) + len(lst2))
    return result


def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def idx_weight(i, mode='linear'):
    if mode == 'linear': return 1 / (i + 1)
    if mode == 'exp': return math.exp(-i)
    if mode == 'sigmoid': return _sigmoid(5 - i)  # 5 means we focuse on the top 5
    if mode == 'no_weight': return 1
    if mode == 'log': return 1 / math.log(2 + i)  # i is start from 0


def route_acc(pred, label, top_n):
    assert set(label).issubset(set(pred)), f"error in prediction:{pred}, label:{label}"
    eval_num = min(top_n, len(label))
    pred = pred[:eval_num]
    if not isinstance(pred, list): pred = pred.tolist()
    if not isinstance(label, list): label = label.tolist()
    for i in range(eval_num):# which means the sub route should be totally correct.
        if not pred[i] == label[i]: return 0
    return 1


def location_deviation(pred, label, label_len, mode='square'):
    # label = label[:get_label_len(label)]
    label = label[:label_len]

    n = len(label)
    # get the location in list 1
    idx_1 = [idx for idx, x in enumerate(label)]
    # get the location in list 2
    for i in range(len(label)):
        if label[i] not in pred:
            print(pred)
            print(label)
    idx_2 = [pred.index(x) for x in label]

    # caculate the distance
    idx_diff = [math.fabs(i - j) for i, j in zip(idx_1, idx_2)]
    weights = [idx_weight(idx, 'no_weight') for idx in idx_1]

    result = list(map(lambda x: x ** 2, idx_diff)) if mode == 'square' else idx_diff
    return sum([diff * w for diff, w in zip(result, weights)]) / n

# https://blog.csdn.net/dcrmg/article/details/79228589
# https://github.com/belambert/edit-distance
def edit_distance(pred, label):
    import edit_distance
    assert set(label).issubset(set(pred)), "error in prediction"
    # Focus on the items in label

    if not isinstance(pred, list): pred = pred.tolist()
    if not isinstance(label, list): label = label.tolist()

    try:
         pred = [x for x in pred if x in label]
         ed = edit_distance.SequenceMatcher(pred, label).distance()
    except:
           print('pred in function:', pred, f'type of pred: {type(pred)}')
           print('label in function:', label, f'type label:{type(label)}')
    return ed


from typing import Dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric(object):
    def __init__(
            self,
            method,
            length_range =  [0, 5],
            max_seq_len = 25,

    ):
        self.max_seq_len = max_seq_len
        self.hr = [AverageMeter() for _ in range(self.max_seq_len)]
        self.lsd = AverageMeter()
        self.krc = AverageMeter()
        self.lmd = AverageMeter()
        self.ed = AverageMeter() #edit distance
        self.acc = [AverageMeter() for _ in range(self.max_seq_len)]
        self.method = method
        self.len_range = length_range



    def update(self, prediction, label, label_len, input_len)->None:
        def tensor2lst(x):
            try:
                return x.cpu().numpy().tolist()
            except:
                return x

        if (self.method == 'torch') or (self.method == 'gcnru'):
            prediction, label, label_len, input_len = [tensor2lst(x) for x in [prediction, label, label_len, input_len]]

        def filter_len(prediction, label, label_len, input_len):
            pred_f = []
            label_f = []
            label_len_f = []
            input_len_f = []
            for i in range(len(label_len)):
                if self.len_range[0] <= label_len[i] <= self.len_range[1]:
                    pred_f.append(prediction[i])
                    label_f.append(label[i])
                    label_len_f.append(label_len[i])
                    input_len_f.append(input_len[i])
            return pred_f, label_f, label_len_f, input_len_f

        # process the prediction
        prediction, label, label_len, input_len = filter_len(prediction, label, label_len, input_len)

        pred = []
        for p, inp_len in zip(prediction, input_len):
            # input = set(range(inp_len))
            input = set([x for x in p if x < len(prediction[0]) - 1])
            tmp = list(filter(lambda pi: pi in input, p))
            pred.append(tmp)

        batch_size = len(pred)

        #Hit Rate
        for n in range(self.max_seq_len):
            hr_n =  np.array([hit_rate(pre, lab, lab_len, n+1) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
            self.hr[n].update(hr_n, batch_size)

        krc =  np.array([kendall_rank_correlation(pre, lab, lab_len) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.krc.update(krc, batch_size)

        lsd = np.array([location_deviation(pre, lab, lab_len, 'square') for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.lsd.update(lsd, batch_size)

        lmd = np.array([location_deviation(pre, lab, lab_len, 'mean') for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.lmd.update(lmd, batch_size)

        ed = np.array([edit_distance(pre, lab[:lab_len]) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
        self.ed.update(ed, batch_size)

        # ACC
        for n in range(self.max_seq_len):
            acc_n = np.array([route_acc(pre, lab[:lab_len], n + 1) for pre, lab, lab_len in zip(pred, label, label_len)]).mean()
            self.acc[n].update(acc_n, batch_size)


    def to_dict(
            self
    ) -> Dict:
        result = {f'hr@{i + 1}': self.hr[i].avg for i in range(10)}
        result.update({f'acc@{i + 1}': self.acc[i].avg for i in range(10)})
        result.update({'lsd': self.lsd.avg, 'lmd': self.lmd.avg, 'krc': self.krc.avg, 'ed': self.ed.avg})
        return result
        # return {'hr': self.hr.tolist(), 'lsd': self.lsd, 'lmd': self.lmd, 'krc': self.krc}

    def to_str(self):
        hr = [round(x.avg, 3) for x in self.hr]
        acc = [round(x.avg, 3) for x in self.acc]
        krc = round(self.krc.avg, 3)
        lsd = round(self.lsd.avg, 3)
        ed = round(self.ed.avg, 3)
        return f'krc:{krc} | lsd:{lsd} | ed:{ed} | hr@1:{hr[0]} | hr@2:{hr[1]} | hr@3:{hr[2]} | acc@1:{acc[0]} | acc@2:{acc[1]} | acc@3:{acc[2]} |'

        # return f'krc:{krc} | lsd:{lsd} | hr@1:{hr[0]} | hr@2:{hr[1]} | hr@3:{hr[2]} '

if __name__ == '__main__':
    for i in range(10):
        pred = [i for i in range(6)]
        np.random.shuffle(pred)
        label = [i for i in range(4)]
        np.random.shuffle(label)
        print('pred:', pred)
        print('label:', label)
        distance = edit_distance(pred, label)

        print('distance:', distance)
        print('-' * 50)

    pass
