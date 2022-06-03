from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import math

class Attention(nn.Module):
    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):

        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)
        e = self.project_ref(ref)

        expanded_q = q.repeat(1, 1, e.size(2))

        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 cou_embed_dim,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 ):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = 'greedy'

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)
        self.merge_linear = nn.Linear(self.hidden_dim + cou_embed_dim + 3, self.hidden_dim)

    def check_mask(self, mask_):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true  # for finished routes
            return mask.masked_fill(mask_mask, False)
        return mask_modify(mask_)

    def update_mask(self, mask, selected):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true # for finished routes
            return mask.masked_fill(mask_mask, False)

        sample_pick = (selected % 2 == 1).nonzero()
        sample_deliver = (selected % 2 == 0).nonzero()

        pick_index = torch.index_select(selected, 0, sample_pick[:, 0])
        deliver_index = torch.index_select(selected, 0, sample_deliver[:, 0])

        pick_index_l = tuple(pick_index.tolist())
        pick_next_deliver_lindex_l = tuple((pick_index + 1).tolist())
        deliver_index_l = tuple(deliver_index.tolist())

        sample_pick_l = tuple(sample_pick.squeeze(1).tolist())
        sample_deliver_l = tuple(sample_deliver.squeeze(1).tolist())

        mask_ = mask.clone()
        mask_[sample_pick_l, pick_index_l] = True
        mask_[sample_pick_l, pick_next_deliver_lindex_l] = False
        mask_[sample_deliver_l, deliver_index_l] = True

        return mask_modify(mask_)


    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context, embed_cou, V_decode_mask):
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        if prev_idxs == None:#first step
            logit_mask = self.check_mask(logit_mask)

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, embed_cou, V_decode_mask, self.mask_glimpses, self.mask_logits)

        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:

            probs[logit_mask] = 0.

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, embed_cou, V_decode_mask, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l =  self.merge_linear(torch.cat([hy, embed_cou], dim=1))

        h_out = (hy, cy)
        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf
                logits[V_decode_mask] = -np.inf #(B, T, N, N)
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf
            logits[V_decode_mask] = -np.inf

        return logits, h_out

    def update_decode_mask(self, idxs, K_mask, E_sd_t_masked, B, i, mask, config):
        E_sd_dif = torch.gather(E_sd_t_masked.permute(1, 0, 2).contiguous(), 0, idxs.view(1, B, 1).expand(1, B, E_sd_t_masked.size()[2])).squeeze(0)
        valid_sample_index = tuple((torch.sum(((mask == False) + 0), dim=1) > config['k_min_nodes']).nonzero().squeeze(1).tolist())#mask when there're at least k unfinished tasks
        if len(valid_sample_index) == 0:
            return K_mask
        else:
            if config['k_nearest neighbors'] == 'n-1':
                max_dis_idx = tuple(torch.argmax(((E_sd_dif * ((mask == False) + 0))[valid_sample_index, :]).squeeze(1), dim=1).tolist())
                K_mask_ = K_mask.clone()
                K_mask_[:, i, :][valid_sample_index, max_dis_idx] = True
            elif config['k_nearest neighbors'] == 'n-2':
                max_dis_idx_1 = tuple(((E_sd_dif * ((mask == False) + 0))[valid_sample_index, :]).topk(2, dim= 1)[1][:, 0].tolist())
                max_dis_idx_2 = tuple(((E_sd_dif * ((mask == False) + 0))[valid_sample_index, :]).topk(2, dim=1)[1][:, 1].tolist())
                K_mask_ = K_mask.clone()
                K_mask_[:, i, :][valid_sample_index, max_dis_idx_1] = True
                K_mask_[:, i, :][valid_sample_index, max_dis_idx_2] = True
            else:
                assert False, "Unknown decode type"

            return K_mask_

    def update_features(self, idxs, node_h, V_val_masked, E_ed_t_masked, E_sd_t_masked, B, embedded_inputs):

        E_ed_t_masked_dif = torch.gather(E_ed_t_masked.permute(1, 0, 2).contiguous(), 0, idxs.
                            view(1, B, 1).expand(1, B, E_ed_t_masked.size()[2])).squeeze(0)
        E_sd_t_masked_dif = torch.gather(E_sd_t_masked.permute(1, 0, 2).contiguous(), 0, idxs.
                            view(1, B, 1).expand(1, B, E_sd_t_masked.size()[2])).squeeze(0)

        V_dy = torch.cat([E_ed_t_masked_dif.unsqueeze(2), E_sd_t_masked_dif.unsqueeze(2)], dim=2)

        context = torch.cat([node_h, V_val_masked, V_dy], dim=2).permute(1, 0, 2).contiguous().clone()

        decoder_input = torch.gather(
            embedded_inputs, #(maxlen, B, H)
            0,
            idxs.contiguous().view(1, B, 1).expand(1, B, embedded_inputs.size()[2])
        ).squeeze(0)

        return decoder_input, context

    def forward(self, decoder_input, embedded_inputs, hidden, context, V_reach_mask_t, node_h,
                V_val_masked, E_ed_t_masked, E_sd_t_masked, embed_cou, start_idx, config):

        B = V_reach_mask_t.size()[0]
        N = V_reach_mask_t.size()[1]
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        mask = Variable(V_reach_mask_t, requires_grad=False)
        idxs = start_idx
        K_mask = torch.BoolTensor(np.full([B, N, N], False)).to(decoder_input.device)#knn mask at each step
        for i in steps:
            K_mask = self.update_decode_mask(idxs, K_mask, E_sd_t_masked, B, i, mask, config)

            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask,
                                                         idxs, i, context, embed_cou, K_mask[:, i, :])
            # select the next inputs for the decoder
            idxs = self.decode(
                probs,
                mask
            )

            decoder_input, context = self.update_features(idxs, node_h, V_val_masked,
                                                          E_ed_t_masked, E_sd_t_masked, B, embedded_inputs)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)

        return (torch.stack(outputs, 1), torch.stack(selections, 1), mask)

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs