from typing import Any, Self, Union

import torch
from einops import rearrange
from scipy.spatial.transform import Rotation

from src.model.flow.fm import FM
from src.model.flow.helpers.igso3_helpers import SampleIGSO3
from src.model.flow.helpers.so3_helpers import rotmat_to_rotvec, rotvec_to_rotmat


Device = Union[str, torch.device]


class SO3FM(FM):
    def __init__(
        self,
        so3_conf: Any,
    ):
        super(FM, self).__init__()
        self.so3_conf = so3_conf
        self._igso3 = None
        self.device = so3_conf.device

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def sampling_from_uniform_so3_distribution(
        self,
        num_batch: int,
        num_res: int,
        device: Device,
    ) -> torch.Tensor:
        rot_mats = Rotation.random(num_batch * num_res).as_matrix()
        rot_mats = torch.tensor(rot_mats, device=device, dtype=torch.float32)
        rot_mats = rot_mats.reshape(num_batch, num_res, 3, 3)
        return rot_mats

    def interpolant(
        self,
        rot_1: torch.Tensor,
        t: torch.Tensor,
        rot_0: torch.Tensor | None = None,
        res_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_batch, num_res = rot_1.shape[:2]
        assert t.shape == torch.Size([num_batch]), f"t should be torch.Size({num_batch})"

        rot_noise = self.igso3.sample(torch.tensor([1.5]), num_batch * num_res)
        rot_noise = rot_noise.reshape(num_batch, num_res, 3, 3).to(rot_1.device)
        rot_0 = torch.einsum("...ij,...jk->...ik", rot_1, rot_noise) if rot_0 is None else rot_0
        train_schedule = self.so3_conf.train_schedule
        t = rearrange(t, "B -> B 1 1")
        if train_schedule == "exp":
            t = torch.exp(-t * self.so3_conf.exp_rate)
        elif train_schedule == "linear":
            t = t
        else:
            raise ValueError(f"Unknown SO3 {train_schedule} Interpolant Schedule.")

        rot_t = self.geodesic_t(t=t, mat=rot_1, base_mat=rot_0)
        identity = torch.eye(3, device=self.device)
        if flow_mask is not None:
            rot_t = self.apply_mask(
                rot_t,
                identity[None, None],
                rearrange(flow_mask, "B N -> B N 1 1"),
            )
        return rot_t

    def sampling(
        self,
        rot_1: torch.Tensor,
        rot_t: torch.Tensor,
        t: torch.Tensor,
        dt: float | torch.Tensor,
        res_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = rearrange(t, " -> 1 1")
        sample_schedule = self.so3_conf.sample_schedule
        if sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif sample_schedule == "exp":
            scaling = self.so3_conf.exp_rate
        else:
            raise ValueError(f"Unknown SO3 {sample_schedule} Sampling Schedule.")
        rot_t_1 = self.geodesic_t(t=dt * scaling, mat=rot_1, base_mat=rot_t)
        return rot_t_1

    def vectorfield(
        self,
        rot_t: torch.Tensor,
        rot_1: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rot_t_minus_1 = torch.einsum(
            "...ij,...jk->...ik",
            rot_t.transpose(-1, -2),
            rot_1,
        )

        u_t = rotmat_to_rotvec(rot_t_minus_1)
        return u_t

    def geodesic_t(
        self: Self,
        t: torch.Tensor,
        mat: torch.Tensor,
        base_mat: torch.Tensor,
        u_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u_t = self.vectorfield(rot_t=base_mat, rot_1=mat, t=t) if u_t is None else u_t
        mat_t = rotvec_to_rotmat(t * u_t)
        if base_mat.shape != mat_t.shape:
            raise ValueError(f"Incompatible shapes: rt={mat_t.shape}, r1={base_mat.shape}")
        mat_t = torch.einsum("...ij,...jk->...ik", base_mat, mat_t)
        return mat_t

    def apply_mask(
        self: Self,
        x_flow: torch.Tensor,
        x_fixed: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = flow_mask * x_flow + (1 - flow_mask) * x_fixed
        return x
