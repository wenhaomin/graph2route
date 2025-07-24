
from config import dict_action, pick_delivery_dict, pick_delivery_info_dict, dict_order
import numpy as np
from sklearn.utils import shuffle
import argparse
import pandas as pd
import math
from geopy.distance import geodesic
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def get_dataset(args):
    fin_list, min_num_nodes, max_num_nodes, pad_value =args['fin_list'], args['min_num'], args['max_num'], args['max_num'] + 1

    N = max_num_nodes + 2
    T = int(max_num_nodes / 2)

    batch_nodes_num = []
    batch_E_ed = []
    batch_V = []
    batch_V_reach_mask = []
    batch_label = []
    batch_label_len = []
    batch_V_ft = []
    batch_V_pt = []
    batch_start_idx = []
    batch_E_sd = []
    batch_V_dt = []
    batch_V_dispatch_mask = []
    batch_E_mask = []
    batch_pt_dif = []
    batch_dt_dif = []
    batch_cou = []
    batch_A = []
    batch_V_num = []

    dis_dict = {}

    for cnt, filepath in enumerate(tqdm(fin_list)):
        if args['is_test'] and cnt > 60: continue
        lines = shuffle(open(filepath, "r").readlines())
        for line_num, line in enumerate(lines):

            line = line.split(" ")

            start_idx = []

            nodes_num = int(line[0]) + 1
            if not (min_num_nodes <= nodes_num <= max_num_nodes):
                continue

            batch_nodes_num.append(nodes_num)
            current_step_index = 0
            start_idx.append(current_step_index)

            sample_current_time = int(line[3])
            nodes_dispatch_time = []
            V_ft = []
            V_pt = []

            for idx in range(1, 5 * nodes_num, 5):
                nodes_dispatch_time.append(int(line[idx + 4]))#有出发点
                V_ft.append(int(line[idx + 3])) #考虑了出发点
                V_pt.append(int(line[idx + 2]))#考虑了出发点

            sorted_nodes_finish_time = sorted(V_ft)[1: ]#label 按顺序取得，去掉出发点
            sorted_dispatch_time = sorted(list(set(nodes_dispatch_time)))
            nodes_finish_label = [int(node) - 1 for node in line[line.index('order') + 2 : line.index('send')]]
            cou = [int(line[line.index('cou') + 1]), int(float(line[line.index('cou') + 2])), float(line[line.index('cou') + 3]), int(float(line[line.index('cou') + 4]))]#id, level, v, maxload

            label = []
            label_len = []
            sample_td = []

            sample_time = sorted_dispatch_time[:]
            sample_time.append(np.inf)
            for t in range(len(sample_time) - 1):
                step_label = []
                step_td = []

                for ft in sorted_nodes_finish_time:#从第一个step开始，第一个接单时间开始，找到下一个接单时间，看这两个时间内有哪些订单完成
                    if ( sample_time[t] < ft <=  sample_time[t + 1]):

                        step_label.append(V_ft.index(ft))#快递员在一个波次中不可能在同一时间处于两个不同的点
                        current_step_index = V_ft.index(ft) #改变当前step时间
                        nodes_finish_label.remove(V_ft.index(ft)) #nodes_finish_label从1开始

                        step_td.append(ft - sample_current_time)
                        sample_current_time = ft

                        #同一时间应该不会完成两个订单，如果在接单时间区间有动作完成，则加入step_label
                        #在最后一个接单时间之后，还会完成部分订单的派送
                #在该时间步遍历完所有样本后，改变当前时间步开始节点索引
                start_idx.append(current_step_index)
                label_len.append(len(step_label))

                for j in range(len(step_label), N):#有一个step_label, 证明走了一步，就有一个time duration
                    step_label.append(pad_value)#对该step进行padding到maxlen
                    step_td.append(0)#step_td pad 0
                label.append(step_label)
                sample_td.append(step_td)
            if nodes_finish_label != []:
                print('wrong!')


            for k in range(len( sorted_dispatch_time) + 1, T + 1):
                label.append([pad_value] * N)
                sample_td.append([0] * N)
                label_len.append(0)
            if np.array(sample_td).any() < 0:
                print('wrong array:', sample_td)
                continue
            for m in range(len(start_idx), T ):
                start_idx.append(current_step_index)


            nodes_coord = []
            V = []
            for idx in range(1, 5 * nodes_num, 5):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
                V.append([float(line[idx]), float(line[idx + 1]),
                                      ((float(line[idx + 2]) - 1.58e9) % (24 * 3600)) / 60])

            #节点球面距离矩阵
            dis_temp_list = []
            E_sd = np.zeros([N, N])
            for i in range(nodes_num):
                for j in range(i + 1, nodes_num):
                    p_i = (nodes_coord[i][1], nodes_coord[i][0])
                    p_j = (nodes_coord[j][1], nodes_coord[j][0])
                    dis_temp = dis_dict.get((p_i, p_j), None)#如果有，得到(p_i, p_j), 否则返回None
                    if dis_temp == None:
                        dis_temp = int(geodesic(p_i,p_j).meters)
                        dis_dict[(p_i, p_j)] = dis_temp

                    dis_temp_list.append(dis_temp)
            E_sd[:nodes_num, :nodes_num] = squareform(dis_temp_list)

            V_reach_mask = np.ones([T , N])
            V_reach_mask[V_reach_mask == 1] = True
            V_num = np.zeros([T, N])
            E_mask = np.zeros([T, N, N])
            V_dispatch_mask = np.zeros([T, N])
            max_dis = 100  # 判别两地是否相同的threshold
            for t in range(len( sorted_dispatch_time)): #T个有效时间步
                dispatched = []
                all_dispatched_index = [] #记录每个step开始时，已经派单的所有订单的索引
                for i in range(1, len(nodes_dispatch_time)):#遍历所有点的接单时间
                    if nodes_dispatch_time[i] <=  sorted_dispatch_time[t]:#在当前时间已经接单
                        dispatched.append(i)#已派单订单索引
                        all_dispatched_index.append(i)#拿到的既有揽收点索引，也有配送点索引，从第一个揽收点开始，不会有出发点
                    if V_ft[i] <=  sorted_dispatch_time[t]:#nodes_dispatch_list 和 V_ft对应
                        dispatched.remove(i)#如果已经完成，则不在下一个step考虑，V_reach_mask仍然为True
                V_reach_mask[t][dispatched] = False
                for k in range(len(V_reach_mask[t]) - 1):
                    if (V_reach_mask[t][k] == False) and (k  % 2 == 1):
                        V_reach_mask[t][k + 1] = True #揽收配送约束

                #构造边和节点的mask
                for node_i in all_dispatched_index:
                    V_dispatch_mask[t][node_i] = 1
                    for node_j in all_dispatched_index:
                        E_mask[t][node_i][node_j] = 1

                if len(dispatched) > 0:  # 不会为0
                    for m in range(len(dispatched)):  # m从0开始
                        V_num[t][dispatched[m]] += 1#当前step中所有已接单未揽收点都为1，如果有相同订单，则后面+1
                        for n in range(m + 1, len(dispatched)):
                            if (dispatched[m] % 2 == 1) and (dispatched[n] % 2 == 1):  # 都是揽收点
                                if E_sd[dispatched[m]][dispatched[n]] <= max_dis:  # 距离在一段范围
                                    V_num[t][dispatched[m]] += 1
                                    V_num[t][dispatched[n]] += 1
                            elif (dispatched[m] % 2 == 0) and (dispatched[n] % 2 == 0):  # 都是配送点
                                if E_sd[dispatched[m]][dispatched[n]] <= max_dis:  # 距离在一段范围
                                    V_num[t][dispatched[m]] += 1
                                    V_num[t][dispatched[n]] += 1


            V_reach_mask= V_reach_mask > 0
            #统一标准化时间，可考虑其他标准化方式
            V_pt = [float(((float(x) - 1.58e9) % (24 * 3600)) / 60) for x in V_pt]
            V_ft = [float(((float(x) - 1.58e9) % (24 * 3600)) / 60) for x in V_ft]
            V_dt = [float(((float(x) - 1.58e9) % (24 * 3600)) / 60) for x in nodes_dispatch_time]
            for idx in range(nodes_num, N):  # 对route超出长度部分padding
                nodes_coord.append([float(0), float(0)])
                V.append([float(0), float(0), float(0)])
                V_pt.append(float(0))
                V_ft.append(float(0))
                V_dt.append(float(0))

            # 节点距离矩阵
            E_ed = squareform(pdist(nodes_coord, metric='euclidean'))
            E_ed[:, nodes_num:] = 0
            E_ed[nodes_num:, :] = 0
            #统一标准化距离，可考虑其他标准化方式
            E_ed = E_ed * 10000  # [N, N] 需要根据当前点索引更新各点的距离特征

            #揽收时间差和接单时间差矩阵
            E_pt_dif = np.zeros([N, N])
            E_dt_dif = np.zeros([N, N])
            for i in range(nodes_num):
                for j in range(nodes_num):
                    E_pt_dif[i][j] = V_pt[i] - V_pt[j]
                    E_dt_dif[i][j] = V_dt[i] - V_dt[j]


            A_reach = ~V_reach_mask + 0
            A = np.zeros([T, N, N])
            for t in range(T):
                cur_idx = start_idx[t]#当前索引
                reachable_nodes = np.argwhere(A_reach[t] == 1).reshape(-1)
                reachable_nodes = np.append(reachable_nodes, [cur_idx])
                for i in range(N):
                    if i in reachable_nodes:
                        for j in range(N):
                            if j in reachable_nodes:
                                if i != j :
                                    A[t][i][j] = 1
                                else:
                                    A[t][i][j] = -1
            # 根据距离最远或待完成时间最长去掉邻接关系
            for t in range(T):
                for i in range(N):
                    if len(np.argwhere(A[t][i] == 1).reshape(-1)) < 4:
                        continue
                    dis_from_i = E_ed[i] * A[t][i]
                    remove_dis_idx = np.argsort(abs(dis_from_i))[-1]
                    time_from_i = E_pt_dif[i] * A[t][i]
                    remove_time_idx = np.argsort(abs(time_from_i))[-1]
                    A[t][i][[remove_dis_idx, remove_time_idx]] = 0

            batch_V_dispatch_mask.append(V_dispatch_mask)
            batch_E_mask.append(E_mask)
            batch_E_ed.append(E_ed)
            batch_E_sd.append(E_sd)
            batch_V.append(V)
            batch_V_reach_mask.append(V_reach_mask)
            batch_label_len.append(label_len)
            batch_label.append(label)
            batch_V_ft.append(V_ft)
            batch_V_pt.append(V_pt)
            batch_start_idx.append(start_idx)
            batch_V_dt.append(V_dt)
            batch_V_num.append(V_num)
            batch_pt_dif.append(E_pt_dif)
            batch_dt_dif.append(E_dt_dif)
            batch_cou.append(cou)
            batch_A.append(A)

    data = {}
    data['nodes_num'] = np.stack(batch_nodes_num, axis=0)
    data['E_ed'] = np.stack(batch_E_ed, axis=0)
    data['E_sd'] = np.stack(batch_E_sd, axis=0)
    data['E_mask'] = np.stack(batch_E_mask, axis=0)
    data['V'] = np.stack(batch_V, axis=0)
    data['V_reach_mask'] = np.stack(batch_V_reach_mask, axis=0)
    data['V_pt'] = np.stack(batch_V_pt, axis=0)
    data['V_ft'] = np.stack(batch_V_ft, axis=0)
    data['V_num'] = np.stack(batch_V_num, axis=0)
    data['V_dispatch_mask'] = np.stack(batch_V_dispatch_mask, axis=0)
    data['V_dt'] = np.stack(batch_V_dt, axis=0)
    data['label_len'] = np.stack(batch_label_len, axis=0)
    data['label'] = np.stack(batch_label, axis=0)
    data['cou'] = np.stack(batch_cou, axis=0)
    data['start_idx'] = np.stack(batch_start_idx, axis=0)
    data['pt_dif'] = np.stack(batch_pt_dif, axis=0)
    data['dt_dif'] = np.stack(batch_dt_dif, axis=0)
    data['A'] = np.stack(batch_A, axis=0)
    return data

def batch_file_name(file_dir, suffix):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
    return L

import os
def get_workspace():

    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)


def is_None(lst:list):
    for x in lst:
        if x is None:
            return True
    return False

# 验证订单时间顺序真实姓
def time_order(lst:list):
    tm = lst[0]
    for x in lst:
        if x < tm:
            return True
        tm = x
    return False

def get_sorted_table(action_data_path, order_data_path):
    action_data = pd.read_csv(action_data_path)
    order_data = pd.read_csv(order_data_path)
    action_dict = {}
    for idx, row in tqdm(action_data.iterrows(), total=action_data.shape[0]):
        order_id = row[dict_action['tracking_id']]
        center_lng, center_lat = row[dict_action['courier_wave_start_lng']], row[dict_action['courier_wave_start_lat']]
        action_type = row[dict_action['action_type']]
        expect_time = row[dict_action['expect_time']]
        if order_id not in action_dict:
            action_dict[order_id] = {
                'center_lng' : center_lng,
                'center_lat' : center_lat,
            }
        if action_type == "PICKUP":
            action_dict[order_id]['leave_time'] = expect_time
        else:
            action_dict[order_id]['finish_time'] = expect_time

    order_data['leave_time'], order_data['finish_time'], order_data['center_lng'], order_data['center_lat'] = '', '', '', ''
    for idx in order_data.index:
        order_id = order_data.loc[idx].values[2]
        order_data.loc[idx, 'leave_time'] = action_dict[order_id]['leave_time']
        order_data.loc[idx, 'finish_time'] = action_dict[order_id]['finish_time']
        order_data.loc[idx, 'center_lng'] = action_dict[order_id]['center_lng']
        order_data.loc[idx, 'center_lat'] = action_dict[order_id]['center_lat']
    return order_data

def get_csv_data(sorted_table, cou_data_path):
    cou_data =  pd.read_csv(cou_data_path)
    cou_data = cou_data.set_index('courier_id', drop=False)
    df = shuffle(sorted_table)
    data_dict = {}
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if idx == 0:
            continue
        order_id = row[dict_order['order_id']]  # 订单id
        rider_id = row[dict_order['rider_id']]  # 骑手id

        dispatch_time = row[dict_order['assigned_time']]  # 派单时间
        shop_time = row[dict_order['estimate_pick_time']]  # 预计出餐时间
        leave_time = row[dict_order['leave_time']]  # 离店时间
        finish_time = row[dict_order['finish_time']]  # 完成订单时间
        promise_time = row[dict_order['promise_deliver_time']]  # 预计完成订单时间

        rider_lng = row[dict_order['center_lng']]  # 骑手位置
        rider_lat = row[dict_order['center_lat']]
        shop_lng = row[dict_order['shop_lng']]  # 商店位置
        shop_lat = row[dict_order['shop_lat']]
        customer_lng = row[dict_order['deliver_lng']]  # 买家位置
        customer_lat = row[dict_order['deliver_lat']]

        wave_idx = row[dict_order['wave_idx']]  # 波次序号
        cou_inf = cou_data.loc[rider_id]
        cou_id = cou_inf[0]
        cou_level = cou_inf[1]
        cou_speed = cou_inf[2]
        cou_load = cou_inf[3]

        if time_order([dispatch_time, leave_time, finish_time]):
            continue

        if is_None(
                [rider_id, dispatch_time, shop_time, leave_time, finish_time, rider_lng, rider_lat, shop_lng, shop_lat,
                 customer_lat, customer_lng, wave_idx]):
            continue

        if math.isnan(float(rider_lat)):
            rider_lat, rider_lng = 0, 0

        key = (rider_id, wave_idx)
        value1 = (leave_time, shop_time, shop_lng, shop_lat, rider_lng, rider_lat, 0, order_id, dispatch_time, cou_id, cou_level, cou_speed, cou_load)
        value2 = (finish_time, promise_time, customer_lng, customer_lat, rider_lng, rider_lat, 1, order_id, dispatch_time)

        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(value1)
        data_dict[key].append(value2)
    return data_dict

class Counter():
    def __init__(self):
        self.id = 0
        self.dict = {}
    def add(self, x):
        if x not in self.dict.keys():
            self.dict[x] = self.id
            self.id += 1
        return self.dict[x]

def make_data(data_dict):
    datas = []
    global cou_new_id
    for k, v in tqdm(data_dict.items()):

        cou_id = counter.add(k[0])

        s, time_array, num, send_array = "", [], len(v), []
        # 数据为字符串 节点个数 节点坐标 商家坐标顺序 买家坐标顺序 订单实际顺序
        s += str(num) + " "
        #各波次订单数几乎小于24
        if num >= 24:
            continue
        get_first_send_array = []
        for i in range(len(v)):
            get_first_send_array.append(v[i][pick_delivery_info_dict['dispatch_time']])
        first_send_time = sorted(get_first_send_array)[0]

        s += str(v[pick_delivery_dict['pick']][pick_delivery_info_dict['rider_lng']]) + " " + str(v[pick_delivery_dict['pick']][pick_delivery_info_dict['rider_lat']]) + " " + str(first_send_time) + " " + str(first_send_time) \
             + " " + str(first_send_time) + " "

        # 节点坐标
        for i in range(len(v)):
            s += str(v[i][pick_delivery_info_dict['shop_or_custormer_lng']]) + " " + str(v[i][pick_delivery_info_dict['shop_or_custormer_lat']]) + " " + str(v[i][pick_delivery_info_dict['shop_or_promise_time']]) + " " + str(v[i][pick_delivery_info_dict['leave_or_finish_time']]) + " " + str(v[i][pick_delivery_info_dict['dispatch_time']]) + " "

            time_array.append(v[i][0])
            send_array.append(v[i][8])
        # 商家坐标顺序
        s += "pick "
        for i in range(2, len(v) + 2, 2):
            s += str(i) + " "
        # 买家坐标顺序
        s += "deliver "
        for i in range(2, len(v) + 2, 2):
            s += str(i + 1) + " "
        # 订单实际顺序
        s += "order "
        a = np.argsort(time_array)
        s += str(1) + " "
        for i in a:
            s += str(i + 2) + " "
        # 订单派单顺序
        s += "send "
        a = np.argsort(send_array)
        for i in a:
            s += str(i + 2) + " "
        s += "cou "
        s += str(cou_id) + " " + str(v[pick_delivery_dict['pick']][pick_delivery_info_dict['courier_level']]) + " " + str(v[pick_delivery_dict['pick']][pick_delivery_info_dict['courier_speed']]) + " "+ str(v[pick_delivery_dict['pick']][pick_delivery_info_dict['courier_load']]) + " "
        datas.append(s)
    return datas

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
def split_data(fout_temp, datas, idxs, mode):
    file_name = fout_temp + f'/{idxs}.txt'
    dir_check(file_name)
    file = open(file_name, 'w')
    schar = '\n'
    if mode == 'train': file.write(schar.join(datas[:len(datas) // 2]))
    elif mode == 'val': file.write(schar.join(datas[len(datas) // 2 + 1:(len(datas) // 4) * 3]))
    else: file.write(schar.join(datas[(len(datas) // 4) * 3:-1]))

def kernel(params={}):
    sorted_table = get_sorted_table(params['action'], params['order'])
    # 打乱数据，得到按波次和骑手划分的dict
    data_dict = get_csv_data(sorted_table, params['cou'])
    # 根据data dict 划分样本
    datas = make_data(data_dict)
    # 根据得到的样本划分训练集、验证集、测试集
    split_data(params['fout_temp'], datas, params['idxs'], params['mode'])

def get_params():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--min_num', type=int, default=0, help = 'minimal number of task')
    parser.add_argument('--max_num',  type=int, default=25, help = 'maxmal number of task, if even, pad to max_num + 2, else pad to max_num + 1')
    parser.add_argument('--task', type=str, default='fd', help='fd in this case')
    parser.add_argument('--method', type=str, default='food', help='food in this case')
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--num_thread', type=int, default=4, help='for multi-thread data preprocess')
    parser.add_argument('--data_start_day', type=int, default=2, help='the load of first day data on my pc is problematic')
    parser.add_argument('--pad_value', type=int, default=26, help='value of padding')
    parser.add_argument('--dataset', default='food_pd', type=str, help='food_pd in this case')
    parser.add_argument('--start_day', type=int, default=2, help='start day of the data (Feb.), start from 2')
    parser.add_argument('--end_day', type=int, default=4, help='end day of the data (Feb.), end at 30')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':

    params = vars(get_params())
    counter = Counter()
    for mode in ['train', 'val', 'test']:
        # data preprocess
        fout = ws + '/data/dataset/' + params['dataset']  + f'/{mode}.npy'
        dir_check(fout)

        temp_file = ws + '/data/temp/' + 'fd' + f'/{mode}' + '_' + params['method'] + '/'
        fin_list_np = batch_file_name(temp_file, '.txt')
        if len(fin_list_np) == 0:
            args_lst = []
            for i in range(params['start_day'], params['end_day']):
                fin_action_file = (ws + '/data/raw_data' + '/' + 'fd' + '/action/action_202002' + str(i) + '.txt')
                fin_order_file = (ws + '/data/raw_data' + '/' + 'fd' + '/order/order_202002' + str(i) + '.txt')
                fin_cou_file = (ws + '/data/raw_data' + '/' + 'fd' + '/courier/courier_202002' + str(i) + '.txt')
                args = {'fout_temp':temp_file, 'action': fin_action_file, 'order':fin_order_file, 'cou': fin_cou_file, 'method':params['method'], 'mode':mode, 'idxs':i}
                args_lst.append(args)
            multi_thread_work(args_lst, kernel, params['num_thread'])

        fin_list_np = batch_file_name(temp_file, '.txt')
        params.update({'fin_list':fin_list_np})
        # make numpy dataset for model to load
        np.save(fout, get_dataset(params))
        print(f'Save:', fout)
        data = np.load(fout, allow_pickle=True).item()