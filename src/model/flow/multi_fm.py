import logging
from copy import deepcopy
from typing import Any, Self, TypeAlias, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import src.data.basics.utils as du
from src.data.basics.all_atom_na import transrot_to_atom37_rna
from src.model.flow.aa_fm import AAFM
from src.model.flow.fm import FM
from src.model.flow.r3_fm import R3FM
from src.model.flow.so3_fm import SO3FM


Device: TypeAlias = Union[str, torch.device]


class MultiFM(FM):
    def __init__(self: Self, fm_conf: Any):
        super(FM, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.fm_conf = fm_conf

        self.device: Device = fm_conf.device
        self.num_tokens: int = fm_conf.num_tokens

        self.flow_rot: bool = fm_conf.flow_rot
        self.flow_trans: bool = fm_conf.flow_trans
        self.flow_aa: bool = fm_conf.flow_aa

        self.so3_fm = SO3FM(fm_conf.so3_conf)
        self.r3_fm = R3FM(fm_conf.r3_conf)
        self.aa_fm = AAFM(fm_conf.aa_conf)

    def sample_t(self, num_batch: int):
        t = torch.rand(num_batch, device=self.device)
        return t * (1 - 2 * self.fm_conf.min_t) + self.fm_conf.min_t

    def sample_rotmat_t(
        self,
        t: torch.Tensor,
        schedule_type: str,
        exp_rate: float | None = None,
    ):
        if schedule_type == "exp":
            return 1 - torch.exp(-t * exp_rate)
        elif schedule_type == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {schedule_type}")

    def interpolant(self: Self, batch) -> Any:
        batch_t = deepcopy(batch)

        # ground_truth
        trans_1 = batch["trans_1"]  # [B, N, 3]
        rotmats_1 = batch["rotmats_1"]  # [B, N, 3, 3]
        aatypes_1 = batch["aatypes_1"]  # [B, N]
        res_mask = batch["res_mask"]  # [B, N]
        flow_mask = batch["flow_mask"]  # [B, N]
        num_batch, num_res = flow_mask.shape

        # [B, 1]
        if self.fm_conf.codesign_separate_t:
            u = torch.rand((num_batch,), device=self.device)
            forward_fold_mask = (u < self.fm_conf.codesign_forward_fold_prop).float()
            inverse_fold_mask = (
                u
                < (
                    self.fm_conf.codesign_inverse_fold_prop
                    + self.fm_conf.codesign_forward_fold_prop
                )
            ).float() * (u >= self.fm_conf.codesign_forward_fold_prop).float()

            normal_structure_t = self.sample_t(num_batch)  # [B]
            inverse_fold_structure_t = torch.ones((num_batch,), device=self.device)
            normal_aa_t = self.sample_t(num_batch)
            forward_fold_aa_t = torch.ones((num_batch,), device=self.device)
            aa_t = forward_fold_mask * forward_fold_aa_t + (1 - forward_fold_mask) * normal_aa_t
            structure_t = (
                inverse_fold_mask * inverse_fold_structure_t
                + (1 - inverse_fold_mask) * normal_structure_t
            )
            so3_t = structure_t
            r3_t = structure_t
            aa_t = aa_t

        else:
            t = self.sample_t(num_batch)
            so3_t = t
            r3_t = t
            aa_t = t

        batch_t["so3_t"] = so3_t
        batch_t["r3_t"] = r3_t
        batch_t["aa_t"] = aa_t

        # apply interpolant
        if self.flow_trans:
            trans_t = self.r3_fm.interpolant(
                x_1=trans_1,
                t=r3_t,
                x_0=None,
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during flowing")
        batch_t["trans_t"] = trans_t

        if self.flow_rot:
            rotmats_t = self.so3_fm.interpolant(
                rot_1=rotmats_1,
                t=so3_t,
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during flowing")
        batch_t["rotmats_t"] = rotmats_t

        if self.flow_aa:
            aatypes_t = self.aa_fm.interpolant(
                x_1=aatypes_1,
                t=aa_t,
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
        else:
            aatypes_t = aatypes_1
        batch_t["aatypes_t"] = aatypes_t

        # sc: self-condition
        batch_t["sc_trans"] = torch.zeros_like(trans_1)
        batch_t["sc_aatypes"] = rearrange(torch.zeros_like(aatypes_1), "... -> ... 1").repeat(
            1, 1, self.num_tokens
        )
        return batch_t

    def sampling(
        self,
        num_batch: int,
        num_res: int,
        model: nn.Module,
        num_timesteps: int | None = None,
        trans_0: torch.Tensor | None = None,
        rotmats_0: torch.Tensor | None = None,
        aatypes_0: torch.Tensor | None = None,
        trans_1: torch.Tensor | None = None,
        rotmats_1: torch.Tensor | None = None,
        aatypes_1: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
        chain_idx: torch.Tensor | None = None,
        res_idx: torch.Tensor | None = None,
        t_nn: nn.Module | None = None,
        lig_feats: dict | None = None,
        forward_folding: bool = False,
        inverse_folding: bool = False,
        separate_t: bool = False,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self.device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = (
                self.r3_fm.sampling_from_centered_gaussian_distribution(
                    num_batch,
                    num_res,
                    self.device,
                )
                * du.NM_TO_ANG_SCALE
            )

        if rotmats_0 is None:
            rotmats_0 = self.so3_fm.sampling_from_uniform_so3_distribution(
                num_batch,
                num_res,
                self.device,
            )

        if aatypes_0 is None:
            if self.fm_conf.aa_conf.interpolant_type == "masking":
                aatypes_0 = self.aa_fm.sampling_from_categorical_distribution(
                    num_batch,
                    num_res,
                    self.device,
                )

            elif self.fm_conf.aa_conf.interpolant_type == "uniform":
                aatypes_0 = self.aa_fm.sampling_from_uniform_distribution(
                    num_batch,
                    num_res,
                    low=0,
                    high=self.num_tokens,
                    device=self.device,
                )
            else:
                raise ValueError(
                    f"Unknown AA interpolant type {self.fm_conf.aa_conf.interpolant_type}"
                )

        if res_idx is None:
            res_idx = torch.arange(num_res, device=self.device, dtype=torch.float32)
            res_idx = rearrange(res_idx, "... -> 1 ...")
            res_idx = res_idx.repeat(num_batch, 1)

        if chain_idx is None:
            chain_idx = res_mask

        if flow_mask is None:
            flow_mask = res_mask

        # self-condition
        trans_sc = torch.zeros(num_batch, num_res, 3, device=self.device)
        sc_aatypes = torch.zeros(num_batch, num_res, self.num_tokens, device=self.device)

        batch = {
            "res_mask": res_mask,
            "flow_mask": flow_mask,
            "chain_idx": chain_idx,
            "res_idx": res_idx,
            "sc_trans": trans_sc,
            "sc_aatypes": sc_aatypes,
        }

        if lig_feats:
            batch.update(lig_feats)

        # setup ground-truth samples
        if trans_1 is None:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self.device)

        if rotmats_1 is None:
            rotmats_1 = rearrange(torch.eye(3, device=self.device), "B N -> 1 1 B N")
            rotmats_1 = rotmats_1.repeat(num_batch, num_res, 1, 1)

        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res), device=self.device).long()

        logits_1 = F.one_hot(aatypes_1, num_classes=self.num_tokens).float()

        if forward_folding:  # design backbone
            assert aatypes_1 is not None
            # assert self.flow_conf.noise == 0
        if forward_folding and separate_t:
            aatypes_0 = aatypes_1
        if inverse_folding:  # design sequence
            assert trans_1 is not None
            assert rotmats_1 is not None
        if inverse_folding and separate_t:
            trans_0 = trans_1
            rotmats_0 = rotmats_1

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self.fm_conf.num_timesteps
        timesteps = torch.linspace(self.fm_conf.min_t, 1.0, num_timesteps).to(self.device)
        t_1 = timesteps[0]

        # initial features
        trans_t_1, rotmats_t_1, aatypes_t_1 = trans_0, rotmats_0, aatypes_0

        noise_traj = [(trans_t_1.detach().cpu(), rotmats_t_1.detach().cpu())]
        noise_types = [aatypes_t_1.detach().cpu()]

        clean_traj = []
        clean_types = []

        for t_2 in timesteps[1:]:
            # Run model.
            if self.flow_trans:
                batch["trans_t"] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError("Must provide trans_1 if not flowing.")
                batch["trans_t"] = trans_1

            if self.flow_rot:
                batch["rotmats_t"] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError("Must provide rotmats_1 if not flowing.")
                batch["rotmats_t"] = rotmats_1

            if self.flow_aa:
                batch["aatypes_t"] = aatypes_t_1
            else:
                if aatypes_1 is None:
                    raise ValueError("Must provide aatype if not flowing.")
                batch["aatypes_t"] = aatypes_1

            t = torch.ones((num_batch), device=self.device) * t_1

            if t_nn is not None:
                batch["r3_t"], batch["so3_t"], batch["aa_t"] = torch.split(t_nn(t), -1)
            else:
                if self.fm_conf.so3_conf.resample_rotmat_t:  # provide_kappa
                    batch["so3_t"] = self.sample_rotmat_t(t, self.fm_conf.so3_conf.exp_rate)
                else:
                    batch["so3_t"] = t
                batch["r3_t"] = t
                batch["aa_t"] = t
            if forward_folding and separate_t:
                batch["aa_t"] = (1 - self.fm_conf.min_t) * torch.ones_like(batch["aa_t"])
            if inverse_folding and separate_t:
                batch["r3_t"] = (1 - self.fm_conf.min_t) * torch.ones_like(batch["r3_t"])
                batch["so3_t"] = (1 - self.fm_conf.min_t) * torch.ones_like(batch["so3_t"])

            d_t = t_2 - t_1

            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            # These are model's predictions for state at t=1 (end state)
            pred_trans_1 = model_out["pred_trans"]  # Predicted coordinates at t=1
            pred_rotmats_1 = model_out["pred_rotmats"]  # Predicted rotation matrices at t=1
            pred_torsions_1 = model_out["pred_torsions"]
            pred_aatypes_1 = model_out["pred_aatypes"]  # Predicted amino acid types at t=1
            pred_logits_1 = model_out["pred_logits"]

            # Record model's clean predictions (t=1) at current timestep
            clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
            clean_types.append(pred_aatypes_1.detach().cpu())

            if forward_folding:
                pred_logits_1 = 100.0 * logits_1
            if inverse_folding:
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1

            if self.fm_conf.self_condition:
                batch["sc_trans"] = self.apply_mask(
                    pred_trans_1,
                    trans_1,
                    flow_mask.unsqueeze(-1),
                )

                if forward_folding:
                    batch["sc_aatypes"] = logits_1
                else:
                    batch["sc_aatypes"] = self.apply_mask(
                        pred_logits_1,
                        logits_1,
                        flow_mask.unsqueeze(-1),
                    )

            # Take reverse step using Euler method
            # Sample intermediate states from t_1 to t_2 using numerical integration
            trans_t_2 = self.r3_fm.sampling(pred_trans_1, trans_t_1, t_1, d_t)
            rotmats_t_2 = self.so3_fm.sampling(pred_rotmats_1, rotmats_t_1, t_1, d_t)

            if self.fm_conf.aa_conf.do_purity:
                aatypes_t_2 = self.aa_fm.sampling_purity(pred_logits_1, aatypes_t_1, t_1, d_t)
            else:
                aatypes_t_2 = self.aa_fm.sampling(pred_logits_1, aatypes_t_1, t_1, d_t)

            trans_t_2 = self.apply_mask(
                trans_t_2,
                trans_1,
                rearrange(flow_mask, "B N -> B N 1"),
            )
            rotmats_t_2 = self.apply_mask(
                rotmats_t_2,
                rotmats_1,
                rearrange(flow_mask, "B N -> B N 1 1"),
            )
            aatypes_t_2 = self.apply_mask(
                aatypes_t_2,
                aatypes_1,
                flow_mask,
            )

            # Update current state for next step
            trans_t_1, rotmats_t_1, aatypes_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2

            # Record the actual sampling trajectory
            noise_traj.append((trans_t_2.detach().cpu(), rotmats_t_2.detach().cpu()))
            noise_types.append(aatypes_t_2.detach().cpu())

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = timesteps[-1]

        if self.flow_trans:
            batch["trans_t"] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError("Must provide trans_1 if not flowing.")
            batch["trans_t"] = trans_1

        if self.flow_rot:
            batch["rotmats_t"] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError("Must provide rotmats_1 if not flowing.")
            batch["rotmats_t"] = rotmats_1

        if self.flow_aa:
            batch["aatype_t"] = aatypes_t_1
        else:
            if aatypes_1 is None:
                raise ValueError("Must provide aatype if not flowing.")
            batch["aatype_t"] = aatypes_1

        with torch.no_grad():
            model_out = model(batch)

        pred_trans_1 = model_out["pred_trans"]
        pred_rotmats_1 = model_out["pred_rotmats"]
        pred_aatypes_1 = model_out["pred_aatypes"]
        pred_torsions_1 = model_out["pred_torsions"]

        if forward_folding:
            pred_aatypes_1 = aatypes_1
        if inverse_folding:
            pred_trans_1 = trans_1
            pred_rotmats_1 = rotmats_1

        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        clean_types.append(pred_aatypes_1.detach().cpu())

        noise_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        noise_types.append(pred_aatypes_1.detach().cpu())

        is_na_residue_mask = (
            torch.ones(num_batch, num_res, device=self.device).bool().detach().cpu()
        )
        assert res_mask.shape == is_na_residue_mask.shape, "Shape mismatch between NA masks"
        pred_torsions = pred_torsions_1.detach().cpu()
        # retrieve all-atom backbone in ATOM37 format
        noise_traj_rna = transrot_to_atom37_rna(noise_traj, is_na_residue_mask, pred_torsions)
        noise_traj_rna = [noise_traj_rna, noise_types]
        clean_traj_rna = transrot_to_atom37_rna(clean_traj, is_na_residue_mask, pred_torsions)
        clean_traj_rna = [clean_traj_rna, clean_types]

        return noise_traj_rna, clean_traj_rna

    def apply_mask(
        self: Self,
        x_flow: torch.Tensor,
        x_fixed: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = flow_mask * x_flow + (1 - flow_mask) * x_fixed
        return x
