from typing import Optional
from itertools import chain
from functools import partial
from torch_geometric.utils import to_undirected, dropout_edge
import dgl

import torch
import torch.nn as nn

from gin import GIN
from loss_func import sce_loss, calc_loss, cos_loss



class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            activation: str,
            dropout: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            drop_p: float = 0.3,
            alpha: float = 2,
            tau: float = 2,
            concat_hidden: bool = False
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._p = drop_p
        self._t = tau
        self._alpha = alpha

        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        

        # build encoder
        self.encoder = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=num_hidden,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=True,
        )
       
        # build decoder for attribute prediction
        self.decoder = GIN(
            in_dim=num_hidden,
            num_hidden=num_hidden,
            out_dim=in_dim,
            num_layers=1,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=False
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(num_hidden * num_layers, num_hidden, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)

        # * setup loss function
        self.criterion_rec = sce_loss # 重建损失
        self.criterion_con = calc_loss # 对比损失


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha)
        elif loss_fn == "cos":
            criterion = partial(cos_loss, t=alpha)
        else:
            raise NotImplementedError
        return criterion


    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.target.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
        
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        # num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        # token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[mask_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def encoding_drop_edges(self, g, x, p=0.3):
        if p < 0. or p > 1.:
            raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
        # use_g = g.clone()
        src, tgt = g.remove_self_loop().edges()
        edge_index = torch.vstack((src, tgt))
        edge_index = to_undirected(edge_index)

        edge_index, edge_id = dropout_edge(edge_index=edge_index, p=p, force_undirected=True)

        use_g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
        # use_g =  dgl.to_bidirected(use_g.cpu())
        use_g = use_g.remove_self_loop().add_self_loop()
        
        use_x = x.clone()

        return use_g, use_x


    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g, x):
        mask_nodes_use_g, mask_nodes_use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        drop_edges_use_g, drop_edges_use_x = self.encoding_drop_edges(g, x, self._p)

        mask_nodes_enc_rep, mask_nodes_all_hidden = self.encoder(mask_nodes_use_g, mask_nodes_use_x, return_hidden=True)
        drop_edges_enc_rep, drop_edges_all_hidden = self.encoder(drop_edges_use_g, drop_edges_use_x, return_hidden=True)

        if self._concat_hidden:
            mask_nodes_enc_rep = torch.cat(mask_nodes_all_hidden, dim=1)
            drop_edges_enc_rep = torch.cat(drop_edges_all_hidden, dim=1)


        mask_nodes_rec_rep = mask_nodes_enc_rep[mask_nodes]
        drop_edges_rec_rep = drop_edges_enc_rep[mask_nodes]
        loss_con = self.criterion_con(mask_nodes_rec_rep, drop_edges_rec_rep, temperature=self._t)

        sum_rep = mask_nodes_enc_rep + drop_edges_enc_rep

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(sum_rep)

        recon = self.decoder(mask_nodes_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss_rec = self.criterion_rec(x_rec, x_init, alpha=self._alpha)

        loss = loss_rec + loss_con

        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    