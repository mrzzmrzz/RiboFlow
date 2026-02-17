import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.data.basics import all_atom_na
from src.data.basics.constants import nucleotide_constants
from src.model.eval import metrics
from src.model.eval import utils as eu
from src.model.flow.multi_fm import MultiFM
from src.model.flow_model import FlowModel
from src.model.utils import stack_batch_elements


class FlowModule(nn.Module):
    def __init__(self, flow_module_conf, logger=None):
        super().__init__()
        self.flow_module_conf = flow_module_conf
        self.model = FlowModel(flow_module_conf.flow_model_conf)
        self.flowmatcher = MultiFM(flow_module_conf.fm_conf)
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.step = 0

    def model_step(self, noisy_batch: Any):
        train_conf = self.flow_module_conf.train_conf
        loss_mask = noisy_batch["res_mask"] * noisy_batch["flow_mask"]
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        bb_mask = noisy_batch["res_mask"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)
        num_batch, num_res = loss_mask.shape
        torsions_start_index = 0
        torsions_end_index = 8
        num_torsions = torsions_end_index - torsions_start_index
        if train_conf.num_non_frame_atoms == 0:
            # [C3', C4', O4']
            bb_filtered_atom_idx = [2, 3, 6]
        elif train_conf.num_non_frame_atoms == 3:
            # [C3', C4', O4'] + [C1', O3', P]
            bb_filtered_atom_idx = [2, 3, 6] + [0, 7, 9]
        elif train_conf.num_non_frame_atoms == 7:
            # [C3', C4', O4'] + [C1', C5', O3', P, OP1, OP2, N1]
            bb_filtered_atom_idx = [2, 3, 6] + [0, 4, 7, 9, 10, 11, 12]
        else:
            # default [C3', C4', O4']
            bb_filtered_atom_idx = [2, 3, 6]
        num_bb_atoms = len(bb_filtered_atom_idx)
        loss_denom = torch.sum(loss_mask, dim=-1) * num_bb_atoms
        rotmats_t = noisy_batch["rotmats_t"]
        gt_trans_1 = noisy_batch["trans_1"].type(torch.float32)
        gt_rotmats_1 = noisy_batch["rotmats_1"].type(torch.float32)
        gt_torsions_1 = noisy_batch["torsion_angles_sin_cos"][
            :, :, torsions_start_index:torsions_end_index, :
        ]
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions * 2)
        gt_rot_vf = self.flowmatcher.so3_fm.vectorfield(rotmats_t, gt_rotmats_1)
        gt_aatypes_1 = noisy_batch["aatypes_1"].type(torch.int)
        r3_t = noisy_batch["r3_t"]
        so3_t = noisy_batch["so3_t"]
        aa_t = noisy_batch["aa_t"]
        r3_norm_scale = 1 - torch.min(
            rearrange(r3_t, "B -> B 1 1"),
            torch.tensor(train_conf.t_normal_clip),
        )
        so3_norm_scale = 1 - torch.min(
            rearrange(so3_t, "B -> B 1 1"),
            torch.tensor(train_conf.t_normal_clip),
        )
        if train_conf.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                rearrange(aa_t, "B -> B 1"),
                torch.tensor(train_conf.t_normal_clip),
            )
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]
        pred_rotmats_1 = model_output["pred_rotmats"]
        pred_torsions_1 = model_output["pred_torsions"].reshape(
            num_batch, num_res, num_torsions * 2
        )
        pred_rots_vf = self.flowmatcher.so3_fm.vectorfield(rotmats_t, pred_rotmats_1)
        pred_logits = model_output["pred_logits"]
        if torch.any(torch.isnan(pred_rots_vf)):
            self.logger.warning("NaN encountered in pred_rots_vf, skipping this batch")
            return {"train_loss": None}
        # follow Yim et al. Fast protein backbone generation with SE(3) flow matching
        # see formula 5 & 6
        # https://github.com/microsoft/protein-frame-flow/blob/f50d8dbbdae827be291e9f73d732b61b195f8816/models/flow_module.py#L138
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * train_conf.trans_scale
        trans_loss = train_conf.translation_loss_weight * torch.sum(
            trans_error**2 * loss_mask[..., None],
            dim=(-1, -2),
        )
        trans_loss /= loss_denom + 1e-10
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = train_conf.rotation_loss_weight * torch.sum(
            rots_vf_error**2 * loss_mask[..., None],
            dim=(-1, -2),
        )
        rots_vf_loss /= loss_denom + 1e-10
        ce_loss = (
            torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, self.flow_module_conf.flow_model_conf.num_aa_types),
                gt_aatypes_1.flatten().long(),
                reduction="none",
            ).reshape(num_batch, num_res)
            / cat_norm_scale
        )
        aatypes_loss_denom = torch.sum(loss_mask, dim=-1)
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / (aatypes_loss_denom + 1e-10)
        aatypes_loss *= train_conf.aatypes_loss_weight
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        torsion_loss_denom = torch.sum(loss_mask, dim=-1) * 8  # 8 torsion angles
        torsion_loss = torch.sum(
            torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2
            * loss_mask[..., None],
            dim=(-1, -2),
        )
        torsion_loss = (
            torsion_loss * train_conf.aux_torsion_loss_weight / (torsion_loss_denom + 1e-10)
        )
        se3_vf_loss = trans_loss + rots_vf_loss
        train_loss = se3_vf_loss + aatypes_loss + torsion_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        total_loss = {
            "train_loss": normalize_loss(train_loss),
            "trans_loss": normalize_loss(trans_loss),
            "rots_vf_loss": normalize_loss(rots_vf_loss),
            "aatypes_loss": normalize_loss(aatypes_loss),
            "torsion_loss": normalize_loss(torsion_loss),
        }
        return total_loss

    def forward(self, batch: Any):
        noisy_batch = self.flowmatcher.interpolant(batch)
        flow_mask = noisy_batch["flow_mask"]
        num_aa_type = self.flow_module_conf.flow_model_conf.num_aa_types
        if self.flow_module_conf.fm_conf.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                pred_trans = model_sc["pred_trans"]
                gt_trans = noisy_batch["trans_1"]
                pred_logits = model_sc["pred_logits"]
                gt_aatypes = noisy_batch["aatypes_1"]
                gt_logits = F.one_hot(gt_aatypes.long(), num_aa_type).float()
                trans_sc = pred_trans * flow_mask[..., None] + gt_trans * (
                    1 - flow_mask[..., None]
                )
                sc_aatypes = pred_logits * flow_mask[..., None] + gt_logits * (
                    1 - flow_mask[..., None]
                )
                noisy_batch["sc_trans"] = trans_sc
                noisy_batch["sc_aatypes"] = sc_aatypes
        batch_losses = self.model_step(noisy_batch)
        return batch_losses

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        saved_dir: Path | str,
        save_traj: bool = True,
    ):
        res_mask = batch["res_mask"]
        is_na_residue_mask = batch["is_na_residue_mask"].bool()
        num_batch = res_mask.shape[0]
        num_res = is_na_residue_mask.sum(dim=-1).max().item()
        noise_traj_rna, clean_traj_rna = self.flowmatcher.sampling(
            num_batch,
            num_res,
            self.model,
            trans_1=batch["trans_1"],
            rotmats_1=batch["rotmats_1"],
            aatypes_1=batch["aatypes_1"],
            flow_mask=batch["flow_mask"],
            res_idx=batch["res_idx"],
        )

        is_na_residue_mask = is_na_residue_mask.detach().cpu().numpy()
        batch_metrics = defaultdict(list)
        samples = noise_traj_rna[0]

        if self.flow_module_conf.fm_conf.flow_aa:
            generated_aatypes = noise_traj_rna[1]
        else:
            generated_aatypes = [None] * num_batch

        for i in range(num_batch):
            output_filepath = os.path.join(saved_dir, f"sample_{i}_idx_{batch_idx}_len_{num_res}")
            if save_traj:
                final_pos = stack_batch_elements(samples, i)
                final_aatypes = None
                if generated_aatypes[i] is not None:
                    final_aatypes = stack_batch_elements(generated_aatypes, i)
                    final_aatypes = final_aatypes.numpy().astype(np.int16) + 25

                saved_path = eu.write_complex_to_pdbs(
                    final_pos.numpy(),
                    output_filepath,
                    restype=final_aatypes,
                    is_na_residue_mask=is_na_residue_mask[i],
                )
                metric_final_pos = final_pos[-1].numpy()
            else:
                final_pos = samples[-1][i]
                final_aatypes = None
                if generated_aatypes is not None:
                    final_aatypes = generated_aatypes[-1][i].numpy().astype(np.int16) + 25
                saved_path = eu.write_complex_to_pdbs(
                    final_pos.numpy(),
                    output_filepath,
                    restype=final_aatypes,
                    is_na_residue_mask=is_na_residue_mask[i],
                )
                metric_final_pos = final_pos.numpy()
            c4_idx = nucleotide_constants.atom_order["C4'"]
            rna_c4_c4_matrics = metrics.calc_rna_c4_c4_metrics(metric_final_pos[:, c4_idx])
            for metric_name, metric_value in rna_c4_c4_matrics.items():
                batch_metrics[metric_name].append(metric_value)
            batch_metrics["save_path"].append(str(saved_path))
        batch_metrics = pd.DataFrame(batch_metrics)
        return batch_metrics

    def eval_step(
        self,
        batch: Any,
        saved_dir: Path | str,
        lig_feats: dict | None = None,
        save_traj: bool = True,
    ):
        num_batch = 1
        num_res = batch["num_res"].item()
        sample_id = batch["sample_id"].item()
        is_na_residue_mask = torch.ones(1, num_res).long().unsqueeze(0)

        sample_dir = os.path.join(saved_dir, f"length_{num_res}")
        os.makedirs(sample_dir, exist_ok=True)
        batch_metrics = defaultdict(list)

        noise_traj_rna, clean_traj_rna = self.flowmatcher.sampling(
            num_batch,
            num_res,
            self.model,
            lig_feats=lig_feats,
        )

        samples = noise_traj_rna[0]
        if self.flow_module_conf.fm_conf.flow_aa:
            generated_aatypes = noise_traj_rna[1]
        else:
            generated_aatypes = None

        if save_traj:
            final_pos = stack_batch_elements(samples, 0)
            final_aatypes = None
            if generated_aatypes is not None:
                final_aatypes = stack_batch_elements(generated_aatypes, 0)
                final_aatypes = final_aatypes.numpy().astype(np.int16) + 25
                final_aatypes = final_aatypes[-1]

            eu.write_complex_to_pdbs(
                final_pos.numpy(),
                os.path.join(sample_dir, f"sample_{sample_id}"),
                restype=final_aatypes,
                is_na_residue_mask=is_na_residue_mask.cpu().numpy(),
            )

        final_pos = samples[-1][-1]
        final_aatypes = None
        if generated_aatypes is not None:
            final_aatypes = generated_aatypes[-1][-1].numpy().astype(np.int16) + 25

        eu.write_complex_to_pdbs(
            final_pos.numpy(),
            os.path.join(sample_dir, f"sample_{sample_id}"),
            restype=final_aatypes,
            is_na_residue_mask=is_na_residue_mask.cpu().numpy(),
        )

        final_pos = final_pos.numpy()
        c4_idx = nucleotide_constants.atom_order["C4'"]
        rna_c4_c4_matrics = metrics.calc_rna_c4_c4_metrics(final_pos[:, c4_idx])
        for metric_name, metric_value in rna_c4_c4_matrics.items():
            batch_metrics[metric_name].append(metric_value)
        batch_metrics = pd.DataFrame(batch_metrics)
        return batch_metrics
