import torch
import torch.nn as nn
import numpy as np
from algorithm.graph2route_logistics.encoder import GCNLayer
from algorithm.graph2route_logistics.decoder import Decoder
import time

class Graph2Route(nn.Module):

    def __init__(self, config):
        super(Graph2Route, self).__init__()
        self.config = config
        self.N = config['max_task_num'] # max nodes
        self.device = config['device']

        # input feature dimension
        self.d_v = config.get('node_fea_dim', 8) # dimension of node feature
        self.d_e = config.get('edge_fea_dim', 5) # dimension of edge feature
        self.d_s = config.get('start_fea_dim', 5) # dimension of start node feature
        self.d_h = config['hidden_size'] # dimension of hidden size
        self.d_w = config['worker_emb_dim'] # dimension of worker embedding


        # feature embedding module
        self.worker_emb = nn.Embedding(config['num_worker_logistics'], self.d_w)
        self.node_emb = nn.Linear(self.d_v, self.d_h, bias=False)
        self.edge_emb = nn.Linear(self.d_e, self.d_h, bias=False)
        self.start_node_emb = nn.Linear(self.d_s, self.d_h + self.d_v)

        # encoding module
        self.gcn_num_layers = config['gcn_num_layers']
        self.gcn_layers = nn.ModuleList([GCNLayer(hidden_dim = self.d_h, aggregation ="mean") for _ in range(self.gcn_num_layers)])
        self.graph_gru = nn.GRU(self.N * self.d_h, self.d_h, batch_first=True)
        self.graph_linear = nn.Linear(self.d_h, self.N * self.d_h)

        # decoding module
        self.decoder = Decoder(
            self.d_h + self.d_v,
            self.d_h + self.d_v,
            tanh_exploration=10,
            use_tanh = True,
            n_glimpses = 1,
            mask_glimpses = True,
            mask_logits = True,
        )

    def forward(self, V, V_reach_mask, V_decode_mask, E_abs_dis, E, E_mask, start_fea, cou_fea):
        """
           Args:
               V: node features, including [dispatch time, coordinates, relative distance, absolute distance,
                promised time - dispatch time, and promised time - current time]. (B, T, N, d_v), here d_v = 8.
               V_reach_mask:  mask for reachability of nodes (B, T, N)
               V_decode_mask: init mask for k-nearest nodes (B, T, N, N)

               E_abs_dis: edge absolute geodesic distance matrix (B, N, N)
               E: masked edge features, include [edge absolute geodesic distance, edge relative geodesic distance, input spatial-temporal adjacent feature,
                                            difference of promised pick-up time between nodes, difference of dispatch_time between nodes] (B, T, N, N, d_e), here d_e = 5
               E_mask: Input edge mask for undispatched nodes (B, T, N, N)

               start_fea: features of start nodes at each step, including dispatch time, coordinates, promised pick-up time, finish time (B, T, d_s), here d_s = 5
               cou_fea: features of couriers including id, work days (B, d_w), here d_w = 2

           :return:
               pointer_log_scores.exp() : rank logits of nodes at each step (B * T, N, N)
               pointer_argmax : decision made at each step (B * T, N)
           """

        B, T, N = V_reach_mask.shape
        d_h, d_v = self.d_h, self.d_v

        # batch input
        b_decoder_input = torch.zeros([B, T, d_h + d_v]).to(self.device)
        b_init_hx = torch.randn(B * T, d_h + d_v).to(self.device)
        b_init_cx = torch.randn(B * T, d_h + d_v).to(self.device)

        b_V_reach_mask = V_reach_mask.reshape(B * T, N)

        b_node_h = torch.zeros([B, T, N, d_h]).to(self.device)
        b_edge_h = torch.zeros([B, T, N, N, d_h]).to(self.device)
        b_masked_E = torch.zeros([B, T, N, N]).to(self.device)
        cou = torch.repeat_interleave(cou_fea.unsqueeze(1), repeats=T, dim = 1).reshape(B * T, -1)#(B * T, 4)
        cou_id = cou[:, 0].long()
        embed_cou = torch.cat([self.worker_emb(cou_id), cou[:, 1].unsqueeze(1)], dim=1)#(B*T, 13)

        for t in range(T):
            E_mask_t = torch.FloatTensor(E_mask[:, t, :, :]).to(V.device)#(B, T, N, N)
            graph_node = V[:, t, :, :]

            graph_edge_abs_dis_t = E_abs_dis * E_mask_t  # (B, N, N)  * (B, N, N)
            b_masked_E[:, t, :, :] = graph_edge_abs_dis_t

            graph_node_t = self.node_emb(graph_node)  # B * N * H
            graph_edge_t = self.edge_emb(E[:, t, :, :])
            b_node_h[:, t, :] = graph_node_t
            b_edge_h[:, t, :, :] = graph_edge_t

            decoder_input = self.start_node_emb(start_fea[:, t, :])

            b_decoder_input[:, t, :] = decoder_input

        for layer in range(self.gcn_num_layers):
            b_node_h, b_edge_h = self.gcn_layers[layer](b_node_h.reshape(B * T, N, d_h),
                                                                b_edge_h.reshape(B * T, N, N, d_h))

        b_node_h, _ = self.graph_gru(b_node_h.reshape(B, T, -1))
        b_node_h = self.graph_linear(b_node_h)#（B, T, N * H)

        b_inputs = torch.cat(
            [b_node_h.reshape(B * T, N, d_h), V.reshape(B * T, N, d_v)], dim=2). \
            permute(1, 0, 2).contiguous().clone()

        b_enc_h = torch.cat(
            [b_node_h.reshape(B * T, N, d_h), V.reshape(B * T, N, d_v),], dim=2). \
            permute(1, 0, 2).contiguous().clone()
        masked_E = b_masked_E.clone()
        masked_E[:, :, :, 0] = 0

        (pointer_log_scores, pointer_argmax, final_step_mask) = \
            self.decoder(
                b_decoder_input.reshape(B * T, d_h + d_v),
                b_inputs.reshape(N, T * B, d_h + d_v),
                (b_init_hx, b_init_cx),
                b_enc_h.reshape(N, T * B, d_h + d_v),
                b_V_reach_mask,  embed_cou, V_decode_mask.reshape(B*T, N, N),
                masked_E.reshape(B * T, N, N), self.config)

        return pointer_log_scores.exp(), pointer_argmax

    def model_file_name(self):
        t = time.time()
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.logistics{t}'
        return file_name

# --Dataset
from torch.utils.data import Dataset
class Graph2RouteDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,  # parameters dict
    ) -> None:
        super().__init__()
        if mode not in ["train", "val", "test"]:  # "validate"
            raise ValueError
        path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):

        return len(self.data['V_len'])

    def __getitem__(self, index):

        E_abs_dis = self.data['E_abs_dis'][index]
        E_dis = self.data['E_dis'][index]
        E_pt_dif = self.data['E_pt_dif'][index]
        E_dt_dif = self.data['E_dt_dif'][index]
        E_mask = self.data['E_mask'][index]

        V = self.data['V'][index]
        V_len = self.data['V_len'][index]
        V_reach_mask = self.data['V_reach_mask'][index]
        V_decode_mask = self.data['V_decode_mask'][index]

        label = self.data['label'][index]
        label_len = self.data['label_len'][index]
        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]
        cou_fea = self.data['cou_fea'][index]

        A = self.data['A'][index]

        return  E_abs_dis, E_dis, E_pt_dif, E_dt_dif, V, V_reach_mask,  \
                E_mask, label, label_len, V_len, start_fea, start_idx, cou_fea, V_decode_mask, A



# ---Log--
from my_utils.utils import save2file_meta
def save2file(params):
    from my_utils.utils import ws
    file_name = ws + f'/output/logistics/{params["model"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',

    ]
    save2file_meta(params,file_name,head)