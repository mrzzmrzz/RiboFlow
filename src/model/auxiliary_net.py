"""
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""

import functools as fn

import torch
import torch.nn as nn

from src.data.basics import all_atom_na
from src.data.basics import utils as du
from src.model import msaformer
from src.model.utils import get_index_embedding, get_timestep_embedding


class BioGraphModule(nn.Module):
    def __init__(self, embed_conf):
        super(BioGraphModule, self).__init__()
        torch.set_default_dtype(torch.float32)

        self.embed_conf = embed_conf

        index_embed_size = self.embed_conf.index_embed_size
        node_embed_size = self.embed_conf.node_embed_size
        edge_embed_size = self.embed_conf.edge_embed_size

        # Time step embedding
        t_embed_size = index_embed_size

        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        if self.embed_conf.embed_self_conditioning:
            edge_in += self.embed_conf.num_bins

        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self.embed_conf.index_embed_size,
        )

        self.index_embedder = fn.partial(
            get_index_embedding,
            embedding_dim=self.embed_conf.index_embed_size,
        )

    def cross_concat(self, feats, num_batch, num_res):
        feats = torch.cat(
            [
                torch.tile(feats[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(feats[:, None, :, :], (1, num_res, 1, 1)),
            ],
            dim=-1,
        )

        feats = feats.float().reshape([num_batch, num_res**2, -1])
        return feats

    def forward(
        self,
        *,
        seq_idx,
        t,
        fixed_mask,
        self_conditioning_ca,
    ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """

        num_batch, num_res = seq_idx.shape

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        # node_feats: [time] -> t_embed_size + 1
        node_feats = [prot_t_embed]
        # pair_feats: 2 * (t_embed_size + 1)
        pair_feats = [self.cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        # node_feats:[time, index] -> t_embed_size + 1 + t_embed_size
        node_feats.append(self.index_embedder(seq_idx))

        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        # pair_feats: 2 * (t_embed_size + 1) + t_embed_size
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # Self-conditioning distogram.
        if self.embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self.embed_conf.min_bin,
                self.embed_conf.max_bin,
                self.embed_conf.num_bins,
            )

            # pair_feats: 2 * (t_embed_size + 1) + t_embed_size + num_bins
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class MSAModule(nn.Module):
    def __init__(self, msa_conf):
        super(MSAModule, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.msa_conf = msa_conf

        self.msa_encoder = msaformer.MSATransformer(
            vocab_size=self.msa_conf.num_msa_vocab,
            n_layers=self.msa_conf.msa_layers,
            n_heads=self.msa_conf.msa_heads,
            model_depth=self.msa_conf.msa_embed_size,
            ff_depth=self.msa_conf.msa_hidden_size,
            dropout=self.msa_conf.dropout,
        )

        self.col_attn = msaformer.MultiHeadAttention(
            num_heads=self.msa_conf.msa_heads,
            embed_dim=self.msa_conf.msa_embed_size,
        )

        self.row_attn = msaformer.MultiHeadAttention(
            num_heads=self.msa_conf.msa_heads,
            embed_dim=self.msa_conf.msa_embed_size,
        )

    def forward(
        self,
        msa_feature,
        msa_mask=None,
    ):
        bs, n_msa, n_token = msa_feature.size()
        msa_feature = msa_feature.reshape(bs * n_msa, n_token)
        msa_embed = self.msa_encoder(msa_feature).reshape(bs, n_msa, n_token, -1)
        msa_embed = msa_embed.transpose(1, 2).reshape(bs * n_token, n_msa, -1)

        if msa_mask is not None:
            msa_mask = msa_mask.transpose(1, 2).reshape(bs * n_token, n_msa)
        msa_embed = (
            self.col_attn(msa_embed, msa_embed, mask=msa_mask)
            .reshape(bs, n_token, n_msa, -1)
            .transpose(1, 2)
        )
        msa_embed = msa_embed.reshape(bs * n_msa, n_token, -1)

        if msa_mask is not None:
            msa_mask = msa_mask.reshape(bs, n_token, n_msa)
            msa_mask = msa_mask.transpose(1, 2).reshape(bs * n_msa, n_token)
        msa_embed = self.row_attn(msa_embed, msa_embed, mask=msa_mask).reshape(
            bs, n_msa, n_token, -1
        )
        return msa_embed


class DistModule(nn.Module):
    def __init__(self, dist_conf):
        super(DistModule, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.dist_conf = dist_conf

        edge_embed_dim = self.dist_conf.edge_embed_dim
        dist_min = self.dist_conf.bb_ligand_rbf_d_min
        dist_max = self.dist_conf.bb_ligand_rbf_d_max
        num_rbf = self.dist_conf.num_rbf
        mu = torch.linspace(dist_min, dist_max, num_rbf)
        self.mu = mu.reshape([1, 1, 1, -1])
        self.sigma = (dist_max - dist_min) / num_rbf

        self.edge_embedder = nn.Sequential(
            nn.Linear(num_rbf, edge_embed_dim),
            nn.ReLU(),
            nn.Linear(edge_embed_dim, edge_embed_dim),
            nn.ReLU(),
            nn.Linear(edge_embed_dim, edge_embed_dim),
            nn.LayerNorm(edge_embed_dim),
        )

    def coord2dist(self, coord, edge_mask):
        # [B, 1, N, 3] - [B, N, 1, 3] -> [B, N, N, 3] -> [B, N, N]
        radial = torch.sum((coord.unsqueeze(1) - coord.unsqueeze(2)) ** 2, dim=-1)
        # [B, N, N]
        dist = torch.sqrt(radial + 1e-10) * edge_mask
        radial = radial * edge_mask
        return radial, dist

    def rbf(self, dist):
        dist_expand = torch.unsqueeze(dist, -1)
        rbf = torch.exp(-(((dist_expand - self.mu.to(dist.device)) / self.sigma) ** 2))
        return rbf

    def forward(
        self,
        trans_t,
        rotmats_t,
        torsions_t,
        lig_pos,  # [B, L, 3]
        bb_mask,
        bb_ligand_mask,  # [B, N + L]
    ):
        # [B, N, 37, 3]
        # NA atom37 bb order = ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', ...]
        cur_bb_pos = all_atom_na.to_atom37_rna(
            torch.clone(trans_t),
            torch.clone(rotmats_t),
            bb_mask,
            torch.clone(torsions_t),
        )
        cur_bb_pos = cur_bb_pos[:, :, 3]
        cur_bb_pos = cur_bb_pos.to(lig_pos.device)

        # [B, N + L, 3]
        cur_bb_lig_pos = torch.cat([cur_bb_pos, lig_pos], dim=1)
        # [B, 1, N + L] *  [B, N + L, 1] -> [B, N +L, N + L]
        edge_mask = bb_ligand_mask.unsqueeze(dim=1) * bb_ligand_mask.unsqueeze(dim=2)

        # [B, N + L, N + L]
        radial, dist = self.coord2dist(
            coord=cur_bb_lig_pos,
            edge_mask=edge_mask,
        )

        # [B, N + L, N + L, num_rbf] * [B, N + L, N + L, 1] -> [B, N + L, N + L, num_rbf]
        edge_embed = self.rbf(dist) * edge_mask[..., None]
        # [B, N + L, N + L, num_rbf] -> [B, N + L, N + L, edge_embed_size]
        edge_embed = self.edge_embedder(edge_embed.to(torch.float))

        return edge_embed


class MolNet(nn.Module):
    def __init__(self, mol_conf):
        super(MolNet, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.mol_conf = mol_conf

        num_atom_type = self.mol_conf.num_atom_type
        node_embed_dim = self.mol_conf.node_embed_dim
        edge_embed_dim = self.mol_conf.edge_embed_dim

        dist_min = self.mol_conf.ligand_rbf_d_min
        dist_max = self.mol_conf.ligand_rbf_d_max
        num_rbf = self.mol_conf.num_rbf

        mu = torch.linspace(dist_min, dist_max, num_rbf)
        self.mu = mu.reshape([1, 1, 1, -1])
        self.sigma = (dist_max - dist_min) / num_rbf

        self.node_embedder = nn.Sequential(
            nn.Embedding(num_atom_type, node_embed_dim, padding_idx=0),
            nn.SiLU(),
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.LayerNorm(node_embed_dim),
        )

        self.edge_embedder = nn.Sequential(
            nn.Linear(num_rbf + node_embed_dim + node_embed_dim, edge_embed_dim),
            nn.SiLU(),
            nn.Linear(edge_embed_dim, edge_embed_dim),
            nn.SiLU(),
            nn.Linear(edge_embed_dim, edge_embed_dim),
            nn.LayerNorm(edge_embed_dim),
        )

        self.node_aggregator = nn.Sequential(
            nn.Linear(node_embed_dim + edge_embed_dim, node_embed_dim),
            nn.SiLU(),
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.SiLU(),
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.LayerNorm(node_embed_dim),
        )

    def coord2dist(self, coord, edge_mask):
        # coords: [B, N, 3]
        # [B, 1, N, 3] - [B, N, 1, 3] -> [B, N, N, 3] -> [B, N, N]
        radial = torch.sum((coord.unsqueeze(1) - coord.unsqueeze(2)) ** 2, dim=-1)
        # [B, N, N]
        dist = torch.sqrt(radial + 1e-10) * edge_mask
        radial = radial * edge_mask
        return radial, dist

    def rbf(self, dist):
        # dist: [B, N, N]
        # dist_expand: [B, N, N, 1]
        dist_expand = torch.unsqueeze(dist, -1)
        _mu = self.mu.to(dist.device)
        # [B, N, N, num_rbf]
        rbf = torch.exp(-(((dist_expand - _mu) / self.sigma) ** 2))
        return rbf

    def forward(
        self,
        ligand_atom,
        ligand_pos,
        edge_mask,  # [B, N, N]
    ):
        num_batch, num_atom = ligand_atom.shape
        # [B, L, node_embed_size]
        node_embed = self.node_embedder(ligand_atom)

        # [B, L, L]
        _, dist = self.coord2dist(
            coord=ligand_pos,
            edge_mask=edge_mask,
        )

        # [B, N, N, num_rbf] * [B, N, N, 1] -> [B, N, N, num_rbf]
        edge_embed = self.rbf(dist) * edge_mask[..., None]  # num_rbf

        # [B, N, node_embed_size] -> [B, 1, N, node_embed_size] -> [B, N, N, node_embed_size]
        src_node_embed = node_embed.unsqueeze(1).repeat(1, num_atom, 1, 1)

        # [B, N, node_embed_size] -> [B, N, 1, node_embed_size] -> [B, N, N, node_embed_size]
        tar_node_embed = node_embed.unsqueeze(2).repeat(1, 1, num_atom, 1)

        # node_src, node_tar, edge_dist
        # node_embed_size + node_embed_size + num_rbf
        # [B, N, N, 2 * node_embed_size + num_rbf]
        edge_embed = torch.cat([src_node_embed, tar_node_embed, edge_embed], dim=-1)

        # [B, N, N, 2 * node_embed_size + num_rbf] -> [B, N, N, edge_embed_size]
        edge_embed = self.edge_embedder(edge_embed.to(torch.float))

        # (edge_mask[..., None].sum(dim=1)-> [B, N, N, 1] -> [B, N, 1]
        # [B, N, edge_embed_size] / [B, N, 1] -> [B, N, edge_embed_size]
        # [B, N, edge_embed_size] * [B, N, 1] -> [B, N, edge_embed_size]
        src_node_agg = (
            edge_embed.sum(dim=1) / (edge_mask[..., None].sum(dim=1) + 1e-10)
        ) * ligand_atom.clamp(max=1.0)[..., None]
        # [B, N, edge_embed_size + node_embed_siz]
        src_node_agg = torch.cat([node_embed, src_node_agg], dim=-1)

        # [B, N, node_embed_size] + [B, N, node_embed_size]
        node_embed = node_embed + self.node_aggregator(src_node_agg)
        return node_embed, edge_embed
