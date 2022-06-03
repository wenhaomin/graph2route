# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import math

"""
Metrics
"""


def hit_rate(pred, label, lab_len, top_n=3):
    """
    calculate Hit-Rate@k (HR@k)
    """
    eval_num = min(top_n, lab_len)
    hit_num = len(set(pred[:eval_num]) & set(label[:eval_num]))
    hit_rate = hit_num / eval_num
    return hit_rate


def kendall_rank_correlation(pred, label, label_len):
    """
    caculate  kendall rank correlation (KRC), note that label set is a subset of pred set
    """
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

    n = len(label)
    # compare list 1: compare items between labels
    lst1 = [(label[i], label[j]) for i in range(n) for j in range(i + 1, n)]
    # compare list 2: compare items between label and pred
    lst2 = [(i, j) for i in label for j in not_in_label]

    try:
        hit_lst = [is_concordant(i, j) for i, j in (lst1 + lst2)]
    except:
        print('[warning]: wrong in calculate KRC')
        return float(1)

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
    """
    calculate ACC@k
    """
    assert set(label).issubset(set(pred)), f"error in prediction:{pred}, label:{label}"
    eval_num = min(top_n, len(label))
    pred = pred[:eval_num]
    if not isinstance(pred, list): pred = pred.tolist()
    if not isinstance(label, list): label = label.tolist()
    for i in range(eval_num):# which means the sub route should be totally correct.
        if not pred[i] == label[i]: return 0
    return 1


def location_deviation(pred, label, label_len, mode='square'):
    """
    calculate LSD / LMD
    mode:
       'square', The Location Square Deviation (LSD)
        else:    The Location Mean Deviation (LMD)
    """
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
    """
    calculate edit distance (ED)
    """
    import edit_distance
    assert set(label).issubset(set(pred)), "error in prediction"
    # Focus on the items in the label
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
            length_range,
            max_seq_len = 25,
    ):
        self.max_seq_len = max_seq_len
        self.hr = [AverageMeter() for _ in range(self.max_seq_len)]
        self.lsd = AverageMeter()
        self.krc = AverageMeter()
        self.lmd = AverageMeter()
        self.ed = AverageMeter() #edit distance
        self.acc = [AverageMeter() for _ in range(self.max_seq_len)]
        self.len_range = length_range

    def filter_len(self, prediction, label, label_len, input_len):
        """
        filter the input data,  only evalution the data within len_range
        """
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

    def update(self, prediction, label, label_len, input_len)->None:
        def tensor2lst(x):
            try:
                return x.cpu().numpy().tolist()
            except:
                return x

        prediction, label, label_len, input_len = [tensor2lst(x) for x in [prediction, label, label_len, input_len]]

        # process the prediction
        prediction, label, label_len, input_len = self.filter_len(prediction, label, label_len, input_len)

        pred = []
        for p, inp_len in zip(prediction, input_len):
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


    def to_dict(self) -> Dict:
        result = {f'hr@{i + 1}': self.hr[i].avg for i in range(10)}
        result.update({f'acc@{i + 1}': self.acc[i].avg for i in range(10)})
        result.update({'lsd': self.lsd.avg, 'lmd': self.lmd.avg, 'krc': self.krc.avg, 'ed': self.ed.avg})
        return result

    def to_str(self):
        hr = [round(x.avg, 3) for x in self.hr]
        acc = [round(x.avg, 3) for x in self.acc]
        krc = round(self.krc.avg, 3)
        lsd = round(self.lsd.avg, 3)
        ed = round(self.ed.avg, 3)
        return f'krc:{krc} | lsd:{lsd} | ed:{ed} | hr@1:{hr[0]} | hr@2:{hr[1]} | hr@3:{hr[2]} | acc@1:{acc[0]} | acc@2:{acc[1]} | acc@3:{acc[2]} |'

if __name__ == '__main__':
    # example
    prediction = [4, 1, 2, 5, 3, 0]
    label = [4,3,5,1,2,3]
    label_len = 4 # The observation of label[:label_len] is not affected by new coming orders.

    print('prediction:', prediction)
    print('label:', label)
    print('label not affected by new orders:', label[:label_len])

    print('HR@3:' , hit_rate(prediction, label, label_len, 3))
    print('KRC:'  , kendall_rank_correlation(prediction, label, label_len))
    print('LSD:'  , location_deviation(prediction, label, label_len, 'square'))
    print('LMD'   , location_deviation(prediction, label, label_len, 'mean'))
    print('ACC@2:', route_acc(prediction, label[:label_len], 2))
    print('ED:'   , edit_distance(prediction, label[:label_len]))