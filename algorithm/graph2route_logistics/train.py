# -*- coding: utf-8 -*-
import os
import argparse

import numpy as np
import torch

from my_utils.eval import *
import torch.nn.functional as F

from tqdm import  tqdm
from my_utils.eval import Metric
from my_utils.utils import  to_device, run, dict_merge
from my_utils.utils import get_nonzeros
from algorithm.graph2route_logistics.graph2route_model import Graph2RouteDataset


def collate_fn(batch):
    return  batch

def process_batch(batch, model, device, pad_vaule):
    E_abs_dis, E_dis, E_pt_dif, E_dt_dif, V, V_reach_mask, E_mask, label, label_len, V_len, start_fea,\
    start_idx, cou_fea, V_decode_mask, A = zip(*batch)

    V = torch.FloatTensor(V).to(device)
    V_reach_mask = torch.BoolTensor(V_reach_mask).to(device)

    label = torch.LongTensor(label).to(device)
    E_abs_dis = torch.FloatTensor(E_abs_dis).to(device)
    E_dis = torch.FloatTensor(E_dis).to(device)
    E_pt_dif = torch.FloatTensor(E_pt_dif).to(device)
    E_dt_dif = torch.FloatTensor(E_dt_dif).to(device)

    start_fea = torch.FloatTensor(start_fea).to(device)
    cou_fea = torch.LongTensor(cou_fea).to(device)
    V_decode_mask = torch.BoolTensor(V_decode_mask).to(device)
    A = torch.FloatTensor(A).to(device)


    pred_scores, pred_pointers = model.forward(V, V_reach_mask, E_abs_dis, E_dis,
                                    E_pt_dif, E_dt_dif, start_fea, np.array(E_mask),  cou_fea, V_decode_mask, A)
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    loss = F.cross_entropy(unrolled, label.view(-1), ignore_index = pad_vaule)
    return pred_pointers, loss

# from gcnru_pd_test.gcn_model_split import  beamsearch_tour_nodes_shortest
def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    model.eval()

    evaluator_1 = Metric([params['eval_start'], params['eval_end_1']])
    evaluator_2 = Metric([params['eval_start'], params['eval_end_2']])

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            E_abs_dis, E_dis, E_pt_dif, E_dt_dif, V, V_reach_mask, E_mask, label, label_len,\
            V_len, start_fea, start_idx, cou_fea, V_decode_mask, A = zip(*batch)

            V = torch.FloatTensor(V).to(device)
            V_reach_mask = torch.BoolTensor(V_reach_mask).to(device)

            label_len = torch.LongTensor(label_len).to(device)
            label = torch.LongTensor(label).to(device)

            E_abs_dis = torch.FloatTensor(E_abs_dis).to(device)
            E_dis = torch.FloatTensor(E_dis).to(device)
            E_pt_dif = torch.FloatTensor(E_pt_dif).to(device)
            E_dt_dif = torch.FloatTensor(E_dt_dif).to(device)
            start_fea = torch.FloatTensor(start_fea).to(device)
            cou_fea = torch.LongTensor(cou_fea).to(device)
            V_decode_mask = torch.BoolTensor(V_decode_mask).to(device)
            A = torch.FloatTensor(A).to(device)

            pred_scores, pred_pointers = model.forward(V, V_reach_mask,  E_abs_dis,
                                E_dis, E_pt_dif, E_dt_dif, start_fea, np.array(E_mask), cou_fea, V_decode_mask, A)

            N = pred_pointers.size(-1)
            pred_len = torch.sum((pred_pointers.reshape(-1, N) < N - 1) + 0, dim=1)

            pred_steps, label_steps, labels_len, preds_len = \
                get_nonzeros(pred_pointers.reshape(-1, N), label.reshape(-1, N),
                             label_len.reshape(-1), pred_len, pad_value)

            evaluator_1.update(pred_steps, label_steps, labels_len, preds_len)
            evaluator_2.update(pred_steps, label_steps, labels_len, preds_len)

        if mode == 'val':
            return evaluator_2

        params_1 = dict_merge([evaluator_1.to_dict(), params])
        params_1['eval_min'] = params['eval_start']
        params_1['eval_max'] = params['eval_end_1']
        save2file(params_1)

        print(evaluator_2.to_str())
        params_2 = dict_merge([evaluator_2.to_dict(), params])
        params_2['eval_min'] = params['eval_start']
        params_2['eval_max'] = params['eval_end_2']
        save2file(params_2)

        return evaluator_2

def main(params):
    params['pad_value'] = params['max_task_num'] - 1
    run(params, Graph2RouteDataset, process_batch, test_model,collate_fn)

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    import time, nni
    import logging

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)

        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise