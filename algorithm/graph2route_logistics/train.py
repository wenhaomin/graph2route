# -*- coding: utf-8 -*-
import os
from my_utils.eval import *
import torch.nn.functional as F
os.environ['MKL_SERVICE_FORCE_INTEL']='1'
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
from tqdm import  tqdm
from my_utils.eval import Metric
from my_utils.utils import run, dict_merge
from my_utils.utils import get_nonzeros
from algorithm.graph2route_logistics.graph2route_model import Graph2RouteDataset

def loss_edges(y_pred_edges, y_edges, edge_cw):
    # Edge loss
    y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.permute(0, 3, 1, 2).contiguous() # B x voc_edges x V x V
    loss_edges = nn.NLLLoss(edge_cw)(y, y_edges)
    return loss_edges

def collate_fn(batch):
    return  batch

def process_batch(batch, model, device, pad_vaule):
    E_abs_dis, E_dis, E_pt_dif, E_dt_dif, V, V_reach_mask, E_mask, label, label_len, start_fea,\
    start_idx, cou_fea, V_decode_mask = zip(*batch)

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

    pred_scores, pred_pointers = model.forward(V, V_reach_mask, E_abs_dis, E_dis,
                                    E_pt_dif, E_dt_dif, start_fea, np.array(E_mask), cou_fea, V_decode_mask)
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    loss = F.cross_entropy(unrolled, label.view(-1), ignore_index = pad_vaule)
    return pred_pointers, loss

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    model.eval()

    evaluator_1 = Metric('torch', [1, 11])
    evaluator_2 = Metric('torch', [1, 25])

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            E_abs_dis, E_dis, E_pt_dif, E_dt_dif, V, V_reach_mask, E_mask, label, label_len,\
             start_fea, start_idx, cou_fea, V_decode_mask = zip(*batch)

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

            pred_scores, pred_pointers = model.forward(V, V_reach_mask, E_abs_dis,
                                E_dis, E_pt_dif, E_dt_dif, start_fea, np.array(E_mask), cou_fea, V_decode_mask)

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
        params_1['eval_min'] = 1
        params_1['eval_max'] = 11
        save2file(params_1)

        print(evaluator_2.to_str())
        params_2 = dict_merge([evaluator_2.to_dict(), params])
        params_2['eval_min'] = 1
        params_2['eval_max'] = 25
        save2file(params_2)

        return evaluator_2

def main(params):
    params['model'] = 'graph2route_logistics'
    params['dataset'] = 'logistics_p'
    params['pad_value'] = params['max_num'] - 1
    run(params, Graph2RouteDataset, process_batch, test_model,collate_fn)

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='graph2route_logistics')# decode mask
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('--gcn_num_layers', type=int, default=3)
    parser.add_argument('--is_eval', type=str, default=True, help='True means load existing model' )
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