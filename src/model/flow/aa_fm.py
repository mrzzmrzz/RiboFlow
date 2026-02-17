from typing import Any, Self, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.distributions.categorical import Categorical

from src.model.flow.fm import FM


Device = Union[str, torch.device]

MASK_TOKEN_INDEX = NUM_TOKEN = 4


class AAFM(FM):
    def __init__(
        self: Self,
        aa_conf: Any,
    ):
        self.aa_conf = aa_conf
        self.device = aa_conf.device

    def sampling_from_categorical_distribution(
        self: Self,
        num_batch: int,
        num_res: int,
        device: Device,
    ) -> torch.Tensor:
        samples = torch.ones(num_batch, num_res, device=device) * MASK_TOKEN_INDEX
        return samples

    def sampling_from_uniform_distribution(
        self: Self,
        num_batch: int,
        num_res: int,
        low: int,
        high: int,
        device: Device,
    ) -> torch.Tensor:
        samples = torch.randint_like(
            torch.ones(num_batch, num_res), low=low, high=high, device=device
        )

        return samples

    def interpolant(
        self: Self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
        res_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_batch, num_res = x_1.shape
        t = rearrange(t, "B -> B 1")
        assert t.shape == (num_batch, 1), "t should be in the format [num_batch, 1]"
        if res_mask is not None:
            assert x_1.shape == (num_batch, num_res)

        if self.aa_conf.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self.device)
            x_t = x_1.clone()
            flowing_mask = u < (1 - t)  # (B, N)
            x_t[flowing_mask] = MASK_TOKEN_INDEX

        elif self.aa_conf.interpolant_type == "uniform":
            u = torch.rand(num_batch, num_res, device=self.device)
            x_t = x_1.clone()
            flowing_mask = u < (1 - t)  # (B, N)
            uniform_sample = torch.randint_like(x_t, low=0, high=NUM_TOKEN)
            x_t[flowing_mask] = uniform_sample[flowing_mask]

        else:
            raise ValueError(f"Unknown AA {self.aa_conf.interpolant_type} interpolant type")

        if res_mask is not None:
            x_t = self.apply_mask(x_t, MASK_TOKEN_INDEX, res_mask)
        return x_t

    def sampling(
        self,
        logit_1: torch.Tensor,
        x_t: torch.Tensor,
        t: float | torch.Tensor,
        dt: float | torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_res, S = logit_1.shape
        assert x_t.shape == (batch_size, num_res)

        if self.aa_conf.interpolant_type == "masking":
            assert S == NUM_TOKEN + 1
            device = logit_1.device
            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[MASK_TOKEN_INDEX] = 1.0
            logit_1[:, :, MASK_TOKEN_INDEX] = -1e9
            pt_x1_probs = F.softmax(logit_1 / self.aa_conf.temp, dim=-1)  # (B, D, S)
            x_t_is_mask = (x_t == MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
            # NOTE: noise = the number of categories that need to be generated 4.0
            step_probs = dt * pt_x1_probs * ((1 + self.aa_conf.noise * t) / (1 - t))  # (B, D, S)
            step_probs += (
                dt * (1 - x_t_is_mask) * mask_one_hot.view(1, 1, -1) * self.aa_conf.noise
            )

        elif self.aa_conf.interpolant_type == "uniform":
            assert S == NUM_TOKEN
            assert x_t.max() < (NUM_TOKEN - 1), (
                "No UNK tokens allowed in the uniform sampling step!"
            )
            device = logit_1.device

            # (B, D, S)
            pt_x1_probs = F.softmax(logit_1 / self.aa_conf.temp, dim=-1)

            # (B, D, 1)
            pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=x_t.long().unsqueeze(-1))
            assert pt_x1_eq_xt_prob.shape == (batch_size, num_res, 1)
            N = self.aa_conf.noise
            step_probs = dt * (
                pt_x1_probs * ((1 + N + N * (S - 1) * t) / (1 - t)) + N * pt_x1_eq_xt_prob
            )

        else:
            raise ValueError(f"Unknown AA {self.aa_conf.interpolant_type} interpolant type")

        step_probs = self.regularize_step_probs(step_probs, x_t)
        x_t_1 = Categorical(step_probs).sample()
        return x_t_1

    def sampling_purity(
        self,
        logit_1: torch.Tensor,
        x_t: torch.Tensor,
        t: float | torch.Tensor,
        dt: float | torch.Tensor,
    ):
        batch_size, num_res, S = logit_1.shape
        assert x_t.shape == (batch_size, num_res)
        assert S == 5
        assert self.aa_conf.interpolant_type == "masking"
        device = logit_1.device

        logits_1_wo_mask = logit_1[:, :, 0:-1]  # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / self.aa_conf.temp, dim=-1)  # (B, D, S-1)
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0]  # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (x_t != MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True)  # (B, D)

        unmask_probs = (dt * ((1 + self.aa_conf.noise * t) / (1 - t)).to(device)).clamp(
            max=1
        )  # scalar

        number_to_unmask = torch.binomial(
            count=torch.count_nonzero(x_t == MASK_TOKEN_INDEX, dim=-1).float(),
            prob=unmask_probs,
        )
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S - 1), num_samples=1).view(
            batch_size, num_res
        )

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = (
            sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        )
        masked_sorted_max_logprobs_idcs = (
            mask1 * sorted_max_logprobs_idcs + (1 - mask1) * inital_val_max_logprob_idcs
        ).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(
            dim=1,
            index=masked_sorted_max_logprobs_idcs,
            src=torch.ones((batch_size, num_res), device=device),
        )
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        x_t = x_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self.device)
        re_mask_mask = (u < dt * self.aa_conf.noise).float()
        x_t = x_t * (1 - re_mask_mask) + MASK_TOKEN_INDEX * re_mask_mask

        return x_t

    def regularize_step_probs(
        self,
        step_probs: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_res, S = step_probs.shape
        assert x_t.shape == (batch_size, num_res)
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        # fmt: off
        step_probs.scatter_(-1, x_t[:, :, None].type(torch.int64), 0.0)
        step_probs.scatter(-1, x_t[:, :, None].type(torch.int64), (1.0 - step_probs.sum(dim=-1, keepdim=True)))
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        return step_probs

    def apply_mask(
        self: Self,
        x_flow: torch.Tensor,
        x_fixed: float | torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = flow_mask * x_flow + (1 - flow_mask) * x_fixed
        return x
