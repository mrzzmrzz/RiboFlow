import torch
from torch import nn

from src.model.utils import get_index_embedding, get_timestep_embedding


class NodeFeatureNet(nn.Module):
    def __init__(self, node_conf):
        super(NodeFeatureNet, self).__init__()
        self.node_conf = node_conf
        self.node_embed_dim = self.node_conf.node_embed_dim
        self.pos_embed_dim = self.node_conf.pos_embed_dim
        self.timestep_embed_dim = self.node_conf.timestep_embed_dim

        embed_size = self.pos_embed_dim + self.timestep_embed_dim * 2 + 1

        # fmt: off
        if self.node_conf.embed_aatype:
            # Always 5 because of 4 residues (AUGC) + 1 for unk
            self.aatype_embedding = nn.Embedding(5, self.node_embed_dim)
            embed_size += (self.node_embed_dim + self.timestep_embed_dim + self.node_conf.num_aa_types)

        self.linear = nn.Sequential(
            nn.Linear(embed_size, self.node_embed_dim),
            nn.ReLU(),
            nn.Linear(self.node_embed_dim, self.node_embed_dim),
            nn.ReLU(),
            nn.Linear(self.node_embed_dim, self.node_embed_dim),
            nn.LayerNorm(self.node_embed_dim),
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_timestep_embedding(
            timesteps[:, 0],
            self.timestep_embed_dim,
            max_positions=2056,
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
        self,
        *,
        r3_t,
        so3_t,
        aa_t,
        res_mask,
        flow_mask,
        pos,
        aatypes,
        sc_aatypes,
    ):
        # [batch_size, num_residues, pos_emb_dim]
        pos_emb = get_index_embedding(pos, self.pos_embed_dim, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        #  [batch_size, num_residues, pos_emb_dim + 2 * timestep_emb_dim + 1]
        input_feats = [
            pos_emb,  # [batch_size, num_residues, pos_emb_dim]
            flow_mask[..., None],  # [batch_size, num_residues, 1]
            self.embed_t(so3_t, res_mask),  # [batch_size, num_residues, timestep_emb_dim]
            self.embed_t(r3_t, res_mask),  # [batch_size, num_residues, timestep_emb_dim]
        ]

        if self.node_conf.embed_aatype:
            # [batch_size, num_residues, pos_emb_dim + 2 * timestep_emb_dim + 1 + node_emb_dim]
            input_feats.append(self.aatype_embedding(aatypes.type(torch.int)))
            # [batch_size, num_residues, pos_emb_dim + 3 * timestep_emb_dim + 1 + node_emb_dim]
            # input_feats.append(self.embed_t(aa_t, res_mask))
            input_feats.append(self.embed_t(aa_t, res_mask))
            # [batch_size, num_residues, pos_emb_dim + 3 * timestep_emb_dim + 1 + node_emb_dim + num_aa_type]
            input_feats.append(sc_aatypes)

        return self.linear(torch.cat(input_feats, dim=-1))
