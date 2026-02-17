# https://github.com/microsoft/protein-frame-flow/blob/main/models/utils.py


import math

import numpy as np
import torch
from torch.nn import functional as F

from src.data.basics import utils as du


def stack_batch_elements(samples, i):
    stacked_tensors = []
    for tensor in samples:
        stacked_tensors.append(tensor[i])
    return torch.stack(stacked_tensors, dim=0)


def outer_sum(x, y):
    # x: (a, b, c)
    # y: (a, b, c)
    # result: (a, b, b c)
    return x[..., None, :] + y[..., None, :, :]


def produce_pair_repr(m, n, padding=False, method="repeat"):
    # (bs,n,d), (bs,m,d) -> (bs, n+m, n+m, d)
    assert method in ["repeat", "outer_sum"]

    bs = m.size(0)
    assert n.size(0) == bs
    m_size = m.size(1)
    n_size = n.size(1)
    n_node = m_size + n_size

    if padding:
        n_node = n_node + 1

    if method == "repeat":
        # (bs, m, d) -> (bs, m+n, d)
        m = F.pad(m, pad=(0, 0, 0, n_node - m_size), value=0)

        # (bs, m+n, d) -> (bs, m+n, m+n, d)
        m = m.unsqueeze(1).repeat(1, n_node, 1, 1)
        n = n.unsqueeze(1).repeat(1, n_size, 1, 1)

    elif method == "outer_sum":
        # (bs, m, d) -> (bs, m, m, d)
        m = outer_sum(m, m)
        n = outer_sum(n, n)

        # (bs, n, n, d) -> (bs, n+m, n+m, d)
        m = F.pad(m, pad=(0, 0, 0, n_node - m_size, 0, n_node - m_size), value=0)

    mask = torch.zeros(bs, n_node, n_node).to(m.device)
    mask[:, :m_size, :m_size] = 1.0

    if padding:
        mask[:, m_size + 1 :, m_size + 1 :] = 1.0
        m[:, m_size + 1 :, m_size + 1 :, :] = n
    else:
        mask[:, m_size:, m_size:] = 1.0
        m[:, m_size:, m_size:, :] = n

    m = m * mask.reshape(bs, n_node, n_node, 1)
    return m


def get_index_embedding(indices, embedding_dim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embedding_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embedding_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embedding_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    embedding = math.log(max_positions) / (half_dim - 1)
    embedding = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embedding
    )
    embedding = timesteps.float()[:, None] * embedding[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        embedding = F.pad(embedding, (0, 1), mode="constant")
    assert embedding.shape == (timesteps.shape[0], embedding_dim)
    return embedding


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses
