from copy import deepcopy
from typing import Any, Self, Union

import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

import src.data.basics.utils as du
from src.model.flow.fm import FM


Device = Union[str, torch.device]


class R3FM(FM):
    def __init__(
        self: Self,
        r3_conf: Any,
    ):
        super(FM, self).__init__()
        self.r3_conf = r3_conf
        self.device = r3_conf.device

    def sampling_from_centered_gaussian_distribution(
        self: Self,
        num_batch: int,
        num_res: int,
        device: Device,
    ) -> torch.Tensor:
        noise = torch.randn(num_batch, num_res, 3, device=device)
        noise = noise - torch.mean(noise, dim=-2, keepdim=True)

        return noise

    def interpolant(
        self: Self,
        *,
        x_1: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor | None,
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_batch, num_res = x_1.shape[:2]
        assert t.shape == torch.Size([num_batch]), f"t should be torch.Size({num_batch})"
        x_0 = (
            self.sampling_from_centered_gaussian_distribution(num_batch, num_res, self.device)
            if x_0 is None
            else x_0
        )
        x_0 = x_0 * du.NM_TO_ANG_SCALE
        if self.r3_conf.batch_ot:
            x_0 = self.batch_ot(x_1, x_0, flow_mask)
        t = rearrange(t, "B -> B 1 1")
        x_t = (1 - t) * x_0 + t * x_1
        if flow_mask is not None:
            x_t = self.apply_mask(x_t, x_1, rearrange(flow_mask, "B N -> B N 1"))
        if res_mask is not None:
            x_t = self.apply_mask(x_t, x_1, rearrange(res_mask, "B N -> B N 1"))
        return x_t

    def batch_ot(
        self: Self,
        x_1: torch.Tensor,
        x_0: torch.Tensor,
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x_1.device

        _x_0 = deepcopy(x_0).cpu()
        _x_1 = deepcopy(x_1).cpu()
        _res_mask = deepcopy(res_mask).cpu()

        num_batch, num_res = _x_0.shape[:2]
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_nm_0 = _x_0[noise_idx]
        batch_nm_1 = _x_1[gt_idx]
        batch_mask = _res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        _x_0 = aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
        _x_0 = _x_0.to(device)
        return _x_0

    def sampling(
        self: Self,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        t: float | torch.Tensor,
        dt: float | torch.Tensor,
        res_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = rearrange(t, " -> 1 1")
        v_t = self.vectorfield(x_t, x_1, t)
        perturb = v_t * dt
        if flow_mask is not None:
            perturb *= flow_mask[..., None]

        x_t_1 = x_t + perturb
        return x_t_1

    def vectorfield(
        self: Self,
        x_t: torch.Tensor,
        x_1: torch.Tensor,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        u_t = (x_1 - x_t) / (1 - t + 1e-10)
        return u_t

    def apply_mask(
        self: Self,
        x_flow: torch.Tensor,
        x_fixed: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = flow_mask * x_flow + (1 - flow_mask) * x_fixed
        return x
