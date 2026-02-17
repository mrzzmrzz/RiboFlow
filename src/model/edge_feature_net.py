# https://github.com/microsoft/protein-frame-flow/blob/main/models/edge_embedder.py

import torch
from torch import nn

from src.data.basics.utils import calc_distogram
from src.model.utils import get_index_embedding


class EdgeFeatureNet(nn.Module):
    def __init__(self, edge_conf):
        super(EdgeFeatureNet, self).__init__()
        self.edge_conf = edge_conf

        self.node_embed_dim = self.edge_conf.node_embed_dim
        self.edge_embed_dim = self.edge_conf.edge_embed_dim
        self.hid_dim = self.edge_conf.hid_dim

        self.linear_node_to_edge = nn.Linear(self.node_embed_dim, self.hid_dim)
        self.linear_relpos = nn.Linear(self.hid_dim, self.hid_dim)
        total_edge_feats = self.hid_dim * 3 + self.edge_conf.num_bins * 2 + 2

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.edge_embed_dim),
            nn.ReLU(),
            nn.Linear(self.edge_embed_dim, self.edge_embed_dim),
            nn.ReLU(),
            nn.Linear(self.edge_embed_dim, self.edge_embed_dim),
            nn.LayerNorm(self.edge_embed_dim),
        )

    def embed_relpos(self, r):
        relative_positions = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(relative_positions, self.hid_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def cross_concat(self, feats, num_batch, num_res):
        # output: [B, N, N, hid_dim * 2]
        feats = torch.cat(
            [
                torch.tile(feats[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(feats[:, None, :, :], (1, num_res, 1, 1)),
            ],
            dim=-1,
        )

        feats = feats.float().reshape([num_batch, num_res, num_res, -1])
        return feats

    def forward(
        self,
        node_features,
        trans,
        sc_trans,
        edge_mask,
        flow_mask,
    ):
        # Input: [num_batch, num_res, node_dim]
        num_batch, num_res, _ = node_features.shape

        # [B, N, hid_dim]
        node_to_edge_features = self.linear_node_to_edge(node_features)

        # [B, N, N, hid_dim * 2]
        pair_node_feats = self.cross_concat(node_to_edge_features, num_batch, num_res)

        # [b, n_res]
        seq_idx = (
            torch.arange(num_res, device=node_features.device).unsqueeze(0).repeat(num_batch, 1)
        )

        # [B, N, N, hid_dim]
        relpos_feats = self.embed_relpos(seq_idx)

        # [B, N, N, num_bins]
        dist_feats = calc_distogram(
            trans, min_bin=1e-3, max_bin=20.0, num_bins=self.edge_conf.num_bins
        )
        # [B, N, N, num_bins]
        sc_feats = calc_distogram(
            sc_trans, min_bin=1e-3, max_bin=20.0, num_bins=self.edge_conf.num_bins
        )
        # [B, N, N, hid_dim * 3 + num_bins * 2]
        all_edge_feats = [pair_node_feats, relpos_feats, dist_feats, sc_feats]

        # [B, N, N, 2]
        flow_feat = self.cross_concat(flow_mask[..., None], num_batch, num_res)

        # [B, N, N, hid_dim * 3 + num_bins * 2 + 2]
        all_edge_feats.append(flow_feat)

        # [B, N, N, edge_embed_dim]
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1).to(torch.float))
        edge_feats *= edge_mask.unsqueeze(-1)
        return edge_feats
