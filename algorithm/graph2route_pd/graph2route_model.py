import torch
import torch.nn as nn
import numpy as np
from algorithm.graph2route_pd.graph2route_layers import GCNLayer
from algorithm.graph2route_pd.step_decode import Decoder
import time

class GCNRU(nn.Module):

    def __init__(self, config):
        super(GCNRU, self).__init__()

        self.batch_size = config['batch_size']
        self.max_nodes = config['max_num'] + 2
        self.node_dim = config.get('node_dim', 8)
        self.dynamic_feature_dim = config.get('dynamic_feature_dim', 2)

        self.gcn_hidden_dim = config['hidden_size']
        self.gru_node_hidden_dim = config['hidden_size']
        self.gru_edge_hidden_dim = config['hidden_size']

        self.gcn_num_layers = config['gcn_num_layers']
        self.aggregation = 'mean'
        self.device = config['device']
        self.start_fea_dim = config.get('start_fea_dim', 5)
        self.num_couv = 1000
        self.cou_embed_dim = config.get('courier_embed_dim', 10)
        self.voc_edges_in = config['voc_edges_in']
        self.config = config

        self.gru_node_linear = nn.Linear(self.gcn_hidden_dim * self.max_nodes, self.gru_node_hidden_dim)
        self.gru_edge_linear = nn.Linear(self.gcn_hidden_dim * self.max_nodes * self.max_nodes,
                                         self.gru_edge_hidden_dim)
        self.cou_embed = nn.Embedding(self.num_couv, self.cou_embed_dim)

        self.nodes_embedding = nn.Linear(self.node_dim, self.gcn_hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(4, self.gcn_hidden_dim, bias=False)

        self.start_embed = nn.Linear(self.start_fea_dim, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim)#cat(node_h, node_fea, node_dynamic)

        gcn_layers = []
        for layer in range(self.gcn_num_layers):
            gcn_layers.append(GCNLayer(self.gcn_hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.graph_gru = nn.GRU(self.max_nodes * self.gcn_hidden_dim, self.gcn_hidden_dim, batch_first=True)
        self.graph_linear = nn.Linear(self.gcn_hidden_dim, self.max_nodes * self.gcn_hidden_dim)
        self.decoder = Decoder(
            self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim,
            self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim ,
            self.cou_embed_dim,
            tanh_exploration=10,
            use_tanh = 10 > 0,
            n_glimpses = 1,
            mask_glimpses = True,
            mask_logits = True,
        )

    def get_init_input(self, t, B, V_val, start_idx_t, V, current_time, V_dt, E_mask_t, V_dispatch_mask, E_ed, E_sd):

        current_step_x_nodes = torch.gather(V.permute(1, 0, 2).contiguous(),
                                            0, start_idx_t.view(1, B, 1).
                                            expand(1, B, V.size()[2])).squeeze(0)
        current_step_dispatch_time = torch.gather(V_dt.unsqueeze(2).permute(1, 0, 2).contiguous(),
                                                  0, start_idx_t.view(1, B, 1)).squeeze(0)

        start_fea = torch.cat([current_step_x_nodes, current_time, current_step_dispatch_time], dim=1)#(B, 5)
        decoder_input = self.start_embed(start_fea)
        # mask
        V_val_masked = V_val * V_dispatch_mask[:, t, :].unsqueeze(2).expand(V_val.size(0), V_val.size(1), V_val.size(2))

        E_ed_t_masked = E_ed * E_mask_t
        E_sd_t_masked = E_sd * E_mask_t

        return decoder_input, V_val_masked, E_ed_t_masked, E_sd_t_masked

    def forward(self, E_ed, V, V_reach_mask, V_pt, E_sd, V_ft, start_idx, V_dt, V_num,
                E_mask, V_dispatch_mask, E_pt_dif, E_dt_dif, cou):

        B, N, H = V_reach_mask.shape[0], self.max_nodes, self.gcn_hidden_dim
        T, node_h, edge_h = int((N - 2) / 2), None, None
        # batch input
        batch_decoder_input = torch.zeros([B, T, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim]).to(
            self.device)
        batch_init_hx = torch.randn(B * T, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim).to(
            self.device)
        batch_init_cx = torch.randn(B * T, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim).to(
            self.device)
        batch_V_reach_mask = V_reach_mask.reshape(B * T, N)
        batch_V_val = torch.zeros([B, T, N, self.node_dim]).to(self.device)
        batch_E_ed_t_masked = torch.zeros([B, T, N, N]).to(self.device)
        batch_E_sd_t_masked = torch.zeros([B, T, N, N]).to(self.device)
        batch_node_h = torch.zeros([B, T, N, self.gcn_hidden_dim]).to(self.device)
        batch_edge_h = torch.zeros([B, T, N, N, self.gcn_hidden_dim]).to(self.device)
        batch_V_dy = torch.zeros([B, T, N, self.dynamic_feature_dim]).to(self.device)
        cou = torch.repeat_interleave(cou.unsqueeze(1), repeats=T, dim = 1).reshape(B * T, -1)
        cou_id = cou[:, 0].long()
        embed_cou = torch.cat([self.cou_embed(cou_id), cou[:, 1].unsqueeze(1), cou[:, 2].unsqueeze(1), cou[:, 3].unsqueeze(1)], dim=1)#(B*T, 23)
        for t in range(T):
            E_mask_t = torch.FloatTensor(E_mask[:, t, :, :]).to(V.device)
            t_c = torch.gather(V_ft.unsqueeze(2).permute(1, 0, 2).contiguous(), 0,
                               start_idx[:, t].view(1, B, 1).expand(1, B, 1)).squeeze(0)
            E_ed_dif = torch.gather(E_ed.permute(1, 0, 2).contiguous(), 0,
                                    start_idx[:, t].view(1, B, 1).expand(1, B, E_ed.size()[2])).squeeze(0)  # (B, N)
            E_sd_dif = torch.gather(E_sd.permute(1, 0, 2).contiguous(), 0,
                                    start_idx[:, t].view(1, B, 1).expand(1, B, E_sd.size()[2])).squeeze(0)  # (B, N)

            E_ed_dif = E_ed_dif * V_dispatch_mask[:, t, :]
            E_sd_dif = E_sd_dif * V_dispatch_mask[:, t, :]

            V_val = torch.cat([V, (V_pt - t_c).unsqueeze(2), (t_c - V_dt).unsqueeze(2),
                               E_ed_dif.unsqueeze(2), E_sd_dif.unsqueeze(2),
                               V_num[:, t, :].unsqueeze(2)], dim=2)

            V_dy = torch.cat([E_ed_dif.unsqueeze(2), E_sd_dif.unsqueeze(2)], dim=2)

            graph_node = V_val * V_dispatch_mask[:, t, :].unsqueeze(2).expand(B, N, self.node_dim)  # V_mask: (B, T, N)
            graph_edge_sd = E_sd * E_mask_t  # (B, N, N)  * (B, N, N)
            graph_edge_ed = E_ed * E_mask_t
            graph_edge_pt = E_pt_dif * E_mask_t
            graph_edge_dt = E_dt_dif * E_mask_t

            graph_node_t = self.nodes_embedding(graph_node)  # B * N * H
            graph_edge_t = self.edges_values_embedding(
                torch.cat([graph_edge_sd.unsqueeze(3), graph_edge_ed.unsqueeze(3),
                           graph_edge_pt.unsqueeze(3), graph_edge_dt.unsqueeze(3)], dim=3))  # B * N * N * H
            batch_node_h[:, t, :] = graph_node_t
            batch_edge_h[:, t, :, :] = graph_edge_t

            decoder_input, V_val_masked, E_ed_t_masked, E_sd_t_masked = \
                self.get_init_input(t, B, V_val, start_idx[:, t],
                                    V, t_c, V_dt, E_mask_t, V_dispatch_mask, E_ed, E_sd)

            batch_decoder_input[:, t, :] = decoder_input#(B, T, N)
            batch_V_val[:, t, :, :] = V_val_masked
            batch_E_ed_t_masked[:, t, :, :] = E_ed_t_masked
            batch_E_sd_t_masked[:, t, :, :] = E_sd_t_masked
            batch_V_dy[:, t, :, :] = V_dy

        for layer in range(self.gcn_num_layers):
            batch_node_h, batch_edge_h = self.gcn_layers[layer](batch_node_h.reshape(B * T, N, H),
                                                                batch_edge_h.reshape(B * T, N, N, H))

        batch_node_h, _ = self.graph_gru(batch_node_h.reshape(B, T, -1))
        batch_node_h = self.graph_linear(batch_node_h)  # （B, T, N * H)

        batch_inputs = torch.cat(
            [batch_node_h.reshape(B * T, N, self.gcn_hidden_dim), batch_V_val.reshape(B * T, N, self.node_dim),
             batch_V_dy.reshape(B * T, N, self.dynamic_feature_dim)], dim=2). \
            permute(1, 0, 2).contiguous().clone()#

        batch_enc_h = torch.cat(
            [batch_node_h.reshape(B * T, N, self.gcn_hidden_dim), batch_V_val.reshape(B * T, N, self.node_dim),
             batch_V_dy.reshape(B * T, N, self.dynamic_feature_dim)], dim=2). \
            permute(1, 0, 2).contiguous().clone()

        (pointer_log_scores, pointer_argmax, final_step_mask) = \
            self.decoder(
                batch_decoder_input.reshape(B * T, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim),#(B*T, H+2)
                batch_inputs.reshape(N, T * B, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim),
                (batch_init_hx, batch_init_cx),
                batch_enc_h.reshape(N, T * B, self.gcn_hidden_dim + self.node_dim + self.dynamic_feature_dim),
                batch_V_reach_mask, batch_node_h.reshape(B * T, N, self.gcn_hidden_dim),
                batch_V_val.reshape(B * T, N, self.node_dim), batch_E_ed_t_masked.reshape(B * T, N, N),
                batch_E_sd_t_masked.reshape(B * T, N, N), embed_cou,  start_idx.reshape(B*T))

        return pointer_log_scores.exp(), pointer_argmax

    def model_file_name(self):
        t = time.time()
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.gcnru-pd_{t}'
        return file_name

from torch.utils.data import Dataset
class GCNRUDataset(Dataset):
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
        return len(self.data['nodes_num'])

    def __getitem__(self, index):

        E_ed = self.data['edges_values'][index]
        V = self.data['nodes_feature'][index]# nodes features

        V_reach_mask = self.data['init_mask_ptr'][index]
        label_len = self.data['sample_label_len'][index]
        label = self.data['sample_label'][index]
        V_pt = self.data['nodes_time_feature'][index]

        V_ft = self.data['finish_time'][index]
        start_idx = self.data['start_time_index'][index]
        E_sd = self.data['square_dis'][index]
        V_dt = self.data['nodes_dispatch_time_feature'][index]

        V_num = self.data['order_num'][index]#num of order at each loc at each step

        E_mask = self.data['A_mask'][index]
        V_dispatch_mask = self.data['V_mask'][index]
        pt_dif = self.data['pt_dif'][index]
        dt_dif = self.data['dt_dif'][index]
        cou = self.data['cou'][index]

        return E_ed, V, V_reach_mask, label_len, label, V_pt, V_ft, start_idx, \
               E_sd, V_dt, V_num, E_mask, V_dispatch_mask, pt_dif, dt_dif, cou

# ---Log--
from my_utils.utils import save2file_meta
def save2file(params):
    from my_utils.utils import ws
    file_name = ws + f'/output/food_pd/{params["model"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_num', 'max_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',

    ]
    save2file_meta(params,file_name,head)