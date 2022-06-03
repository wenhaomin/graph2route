# -*- coding: utf-8 -*-
import os
from tqdm import  tqdm

def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()

def dir_check(path):
    """
    check weather the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

def whether_stop(metric_lst = [], n=2, mode='maximize'):
    """
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    """
    if len(metric_lst) < 1:return False # at least have 1 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx,v in enumerate(metric_lst):
        if v == max_v:max_idx = idx
    return max_idx < len(metric_lst) - n

from multiprocessing import Pool
def multi_thread_work(parameter_queue, function_name, thread_number=5):
    """
    For parallelization
    """
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return  result

class EarlyStop():
    """
    For training process, early stop strategy
    """
    def __init__(self, mode='maximize', patience = 1):
        self.mode = mode
        self.patience =  patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1 # the best epoch
        self.is_best_change = False # whether the best change compare to the last epoch

    def append(self, x):
        """
        append a value, then update corresponding variables
        """
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        """
        return the best metric
        """
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def batch_file_name(file_dir, suffix='.train'):
    """
    Find all files whose suffix is [suffix] in given directory [file_dir]
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L

def dict_merge(dict_list = []):
    """
    merge all the dict in the list
    """
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

def get_dataset_path(params = {}):
    """
    get file path of train, validate and test dataset
    """
    if params['model'] == 'graph2route_pd':
        dataset = 'food_pd'
    else:
        dataset = 'logistics'
    params['dataset'] = dataset
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train.npy'
    val_path = file + f'/val.npy'
    test_path = file + f'/test.npy'
    return train_path, val_path, test_path

def write_list_list(fp, list_, model="a", sep=","):
    dir = os.path.dirname(fp)
    if not os.path.exists(dir): os.makedirs(dir)
    f = open(fp,mode=model, encoding="utf-8")
    count=0
    lines=[]
    for line in list_:
        a_line=""
        for l in line:
            l=str(l)
            a_line=a_line+l+sep
        a_line = a_line.rstrip(sep)
        lines.append(a_line+"\n")
        count=count+1
        if count==10000:
            f.writelines(lines)
            count=0
            lines=[]
    f.writelines(lines)
    f.close()

def save2file_meta(params, file_name, head):
    """
    functions for saving results
    """
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t

    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()
    with open(file_name, "a", newline='\n') as file:  #  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)


#----- Training Utils----------
import argparse
import random, torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader

def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')
    # dataset
    parser.add_argument('--min_task_num', type=int, default=0, help = 'minimal number of task')
    parser.add_argument('--max_task_num',  type=int, default=25, help = 'maxmal number of task')
    parser.add_argument('--dataset', default='logistics', type=str, help='food_pd or logistics')
    parser.add_argument('--pad_value', type=int, default=26, help='food servce, max node num is 25 ')

    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--num_epoch', type=int, default=60, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2022, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop at')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 4)')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')
    parser.add_argument('--courier_embed_dim', type = int, default=10, help = 'embed dim for courier id')
    parser.add_argument('--num_of_couriers in food_pd', type=int, default=920)
    parser.add_argument('--num_of_couriers in logistics', type=int, default=2346)

    #common settings for graph2route model
    parser.add_argument('--node_dim', type=int, default=8)
    parser.add_argument('--voc_edges_in', type=int, default=3)
    parser.add_argument('--voc_edges_out', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('--gcn_num_layers', type=int, default=2)
    parser.add_argument('--k_nearest neighbors', type=str, default='n-1')
    parser.add_argument('--k_min_nodes', type=int, default=3)
    # settings for evaluation
    parser.add_argument('--eval_start', type=int, default=1)
    parser.add_argument('--eval_end_1', type=int, default=11)
    parser.add_argument('--eval_end_2', type=int, default=25)

    return parser

def filter_data(data_dict={}, len_key = 'node_len',  min_len=0, max_len=20):
    '''
    filter data, For dataset
    '''
    new_dic = {}
    keep_idx = [idx for idx, l in enumerate(data_dict[len_key]) if l >= min_len and l <= max_len]
    for k, v in data_dict.items():
        new_dic[k] = [data for idx, data in enumerate(data_dict[k]) if idx in keep_idx]
    return new_dic

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch

import nni, time
def train_val_test(train_loader, val_loader, test_loader, model, device, process_batch, test_model, params, save2file):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
    model_name = model.model_file_name() + f'{time.time()}'
    model_path = ws + f'/data/dataset/{params["dataset"]}/sort_model/{model_name}'
    dir_check(model_path)
    for epoch in range(params['num_epoch']):
        if early_stop.stop_flag: break
        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            ave_loss = None
            model.train()
            for i, batch in enumerate(t):
                pred, loss = process_batch(batch, model, device, params['pad_value'])
                if ave_loss is None:
                    ave_loss = loss.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss.item() / (i + 1)
                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss.item()
                t.set_postfix(**postfix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if params['is_test']: break

        val_result = test_model(model, val_loader, device, params['pad_value'], params, save2file, 'val')
        print('\nval result:', val_result.to_str(), 'Best krc:', round(early_stop.best_metric(), 3), '| Best epoch:', early_stop.best_epoch)
        is_best_change = early_stop.append(val_result.to_dict()['krc'])

        if is_best_change:
            print('value:',val_result.to_dict()['krc'], early_stop.best_metric())
            torch.save(model.state_dict(), model_path)
            print('best model saved')
            print('model path:', model_path)

        if params['is_test']:
            print('model_path:', model_path)
            torch.save(model.state_dict(), model_path)
            print('best model saved !!!')
            break
        nni.report_intermediate_result(val_result.to_dict()['krc'])

    try:
        print('loaded model path:', model_path)
        model.load_state_dict(torch.load(model_path))
        print('best model loaded !!!')
    except:
        print('load best model failed')
    test_result = test_model(model, test_loader, device, params['pad_value'],params, save2file, 'test')
    print('\n-------------------------------------------------------------')
    print('Best epoch: ', early_stop.best_epoch)
    print(f'{params["model"]} Evaluation in test:', test_result.to_str())

    nni.report_final_result(test_result.to_dict()['krc'])
    return params


def get_nonzeros(pred_steps, label_steps, label_len, pred_len, pad_value):
    pred = []
    label = []
    label_len_list = []
    pred_len_list = []
    for i in range(pred_steps.size()[0]):
        #remove samples with no label
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].cpu().numpy().tolist())
            pred.append(pred_steps[i].cpu().numpy().tolist())
            label_len_list.append(label_len[i].cpu().numpy().tolist())
            pred_len_list.append(pred_len[i].cpu().numpy().tolist())
    return torch.LongTensor(pred), torch.LongTensor(label),\
           torch.LongTensor(label_len_list), torch.LongTensor(pred_len_list)

def get_model_function(model):
    import algorithm.graph2route_pd.graph2route_model as graph2route_pd
    import algorithm.graph2route_logistics.graph2route_model as graph2route_logistics

    model_dict = {

        'graph2route_pd':(graph2route_pd.GCNRU, graph2route_pd.save2file),
        'graph2route_logistics':(graph2route_logistics.Graph2Route, graph2route_logistics.save2file),

    }
    model, save2file = model_dict[model]
    return model, save2file

def run(params, DATASET, PROCESS_BATCH, TEST_MODEL, collate_fn = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    params['train_path'], params['val_path'],  params['test_path'] = get_dataset_path(params)
    pprint(params)  # print the parameters

    train_dataset = DATASET(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)

    # train, valid and test
    net_models = ['graph2route_pd', 'graph2route_logistics']
    model, save2file = get_model_function(params['model'])
    model = model(params)

    if params['model'] in net_models:
        result_dict = train_val_test(train_loader, val_loader, test_loader, model, device, PROCESS_BATCH, TEST_MODEL, params, save2file)
    else:
        # one can add non-deepmodel here
        pass

    return params



