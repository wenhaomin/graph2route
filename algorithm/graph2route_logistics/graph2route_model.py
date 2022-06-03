import torch
import torch.nn as nn
import numpy as np
from algorithm.graph2route_logistics.graph2route_layers import GCNLayer
from algorithm.graph2route_logistics.step_decode import Decoder
import time

class Graph2Route(nn.Module):

    def __init__(self, config):
        super(Graph2Route, self).__init__()

        self.batch_size = config['batch_size']
        self.max_nodes = config['max_task_num']
        self.node_dim = config.get('node_dim', 8)
        self.dynamic_feature_dim = config.get('dynamic_feature_dim', 2)
        self.voc_edges_in = config.get('voc_edges_in', 2)
        self.voc_edges_out = config.get('voc_edges_out',2)

        self.hidden_dim = config['hidden_size']
        self.gcn_num_layers = config['gcn_num_layers']
        self.aggregation = 'mean'
        self.device = config['device']
        self.start_fea_dim = config.get('start_fea_dim', 5)
        self.num_couv = config['num_of_couriers in logistics']
        self.cou_embed_dim = config.get('courier_embed_dim', 10)
        self.config = config
        self.cou_embed = nn.Embedding(self.num_couv, self.cou_embed_dim)
        self.tanh_exploration = config.get('tanh_exploration', 10)
        self.n_glimpses = config.get('n_glimpses', 1)
        self.edges_values_dim = config.get('edges_values_dim', 5)

        self.gru_node_linear = nn.Linear(self.hidden_dim * self.max_nodes, self.hidden_dim)
        self.gru_edge_linear = nn.Linear(self.hidden_dim * self.max_nodes * self.max_nodes,
                                         self.hidden_dim)

        self.nodes_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(self.edges_values_dim, self.hidden_dim, bias=False)

        self.start_embed = nn.Linear(self.start_fea_dim, self.hidden_dim + self.node_dim)
        self.config = config

        gcn_layers = []
        for layer in range(self.gcn_num_layers):
            gcn_layers.append(GCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        self.graph_gru = nn.GRU(self.max_nodes * self.hidden_dim, self.hidden_dim, batch_first=True)
        self.graph_linear = nn.Linear(self.hidden_dim, self.max_nodes * self.hidden_dim)
        self.decoder = Decoder(
            self.hidden_dim + self.node_dim,
            self.hidden_dim + self.node_dim,
            tanh_exploration=self.tanh_exploration,
            use_tanh = True,
            n_glimpses = self.n_glimpses,
            mask_glimpses = True,
            mask_logits = True,
        )

    def forward(self, V, V_reach_mask,  E_abs_dis, E_dis, E_pt_dif, E_dt_dif, start_fea, E_mask, cou_fea, V_decode_mask, A):
        """
        Args:
            E_abs_dis: edge absolute geodesic distance matrix (batch_size, num_nodes, num_nodes)
            E_dis: edge relative geodesic distance matrix  (batch_size, num_nodes, num_nodes)
            E_pt_dif: difference of promised pick-up time between nodes (batch_size, num_nodes, num_nodes)
            E_dt_dif: difference of dispatch_time between nodes (batch_size, num_nodes, num_nodes)
            E_mask: Input edge mask for undispatched nodes (batch_size, steps, num_nodes, num_nodes)

            V: node features, including dispatch time, coordinates, relative distance, absolute distance,
             promised time - dispatch time, and promised time - current time. (batch_size, steps, num_nodes, 8)
            V_reach_mask:  mask for reachability of nodes (batch_size, num_steps, num_nodes)
            V_decode_mask: init mask for k-nearest nodes (batch_size, steps, num_nodes, num_nodes)
            A: Input spatial-temporal adjacent feature (batch_size, steps, num_nodes, num_nodes)

            start_fea: features of start nodes at each step, including dispatch time, coordinates, promised pick-up time, finish time (batch_size, steps, 5)
            cou_fea: features of couriers including id, work days (batch_size, 2)

        :return:
            pointer_log_scores.exp() : rank logits of nodes at each step (batch_size * steps, num_nodes, num_nodes)
            pointer_argmax : decision made at each step (batch_size * steps, num_nodes)
        """
        B, N, H = V_reach_mask.shape[0], V_reach_mask.shape[2], self.hidden_dim  # batch size, num nodes, gcn hidden dim
        T, node_h, edge_h = V_reach_mask.shape[1], None, None
        # batch input
        batch_decoder_input = torch.zeros([B, T, self.hidden_dim + self.node_dim]).to(self.device)
        batch_init_hx = torch.randn(B * T, self.hidden_dim + self.node_dim).to(self.device)
        batch_init_cx = torch.randn(B * T, self.hidden_dim + self.node_dim).to(self.device)

        batch_V_reach_mask = V_reach_mask.reshape(B * T, N)

        batch_node_h = torch.zeros([B, T, N, self.hidden_dim]).to(self.device)
        batch_edge_h = torch.zeros([B, T, N, N, self.hidden_dim]).to(self.device)
        batch_masked_E = torch.zeros([B, T, N, N]).to(self.device)
        cou = torch.repeat_interleave(cou_fea.unsqueeze(1), repeats=T, dim = 1).reshape(B * T, -1)#(B * T, 4)
        cou_id = cou[:, 0].long()
        embed_cou = torch.cat([self.cou_embed(cou_id), cou[:, 1].unsqueeze(1)], dim=1)#(B*T, 13)

        for t in range(T):
            E_mask_t = torch.FloatTensor(E_mask[:, t, :, :]).to(V.device)#(B, T, N, N)
            graph_node = V[:, t, :, :]

            graph_edge_abs_dis = E_abs_dis * E_mask_t  # (B, N, N)  * (B, N, N)
            graph_edge_dis = E_dis * E_mask_t
            graph_edge_pt = E_pt_dif * E_mask_t
            graph_edge_dt = E_dt_dif * E_mask_t
            graph_edge_A = A[:, t, :, :] * E_mask_t
            batch_masked_E[:, t, :, :] = graph_edge_abs_dis

            graph_node_t = self.nodes_embedding(graph_node)  # B * N * H
            graph_edge_t = self.edges_values_embedding(
                torch.cat([graph_edge_abs_dis.unsqueeze(3), graph_edge_dis.unsqueeze(3),
                           graph_edge_pt.unsqueeze(3), graph_edge_dt.unsqueeze(3), graph_edge_A.unsqueeze(3)], dim=3))  # B * N * N * H

            batch_node_h[:, t, :] = graph_node_t
            batch_edge_h[:, t, :, :] = graph_edge_t

            decoder_input = self.start_embed(start_fea[:, t, :])

            batch_decoder_input[:, t, :] = decoder_input

        for layer in range(self.gcn_num_layers):
            batch_node_h, batch_edge_h = self.gcn_layers[layer](batch_node_h.reshape(B * T, N, H),
                                                                batch_edge_h.reshape(B * T, N, N, H))

        batch_node_h, _ = self.graph_gru(batch_node_h.reshape(B, T, -1))
        batch_node_h = self.graph_linear(batch_node_h)#（B, T, N * H)

        batch_inputs = torch.cat(
            [batch_node_h.reshape(B * T, N, self.hidden_dim), V.reshape(B * T, N, self.node_dim)], dim=2). \
            permute(1, 0, 2).contiguous().clone()

        batch_enc_h = torch.cat(
            [batch_node_h.reshape(B * T, N, self.hidden_dim), V.reshape(B * T, N, self.node_dim),], dim=2). \
            permute(1, 0, 2).contiguous().clone()
        masked_E = batch_masked_E.clone()
        masked_E[:, :, :, 0] = 0

        (pointer_log_scores, pointer_argmax, final_step_mask) = \
            self.decoder(
                batch_decoder_input.reshape(B * T, self.hidden_dim + self.node_dim),
                batch_inputs.reshape(N, T * B, self.hidden_dim + self.node_dim),
                (batch_init_hx, batch_init_cx),
                batch_enc_h.reshape(N, T * B, self.hidden_dim + self.node_dim),
                batch_V_reach_mask,  embed_cou, V_decode_mask.reshape(B*T, N, N),
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