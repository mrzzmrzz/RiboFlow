import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from src.deps.exper import cuda_utils
from src.model.flow_module import FlowModule


torch.set_float32_matmul_precision("medium")


class Engine:
    def __init__(
        self,
        exp_conf: Any,
    ):
        super().__init__()
        self.exp_conf = exp_conf.exp_conf
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.exp_conf.gradient_accumulation_steps,
        )
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.rank = self.accelerator.process_index
        self.logger = self.setup_logger(self.exp_conf)
        self.logger.info(f"Accelerator initialized. Device: {self.device}")
        self.logger.info("Initializing model and optimizer...")
        self.model = FlowModule(self.exp_conf.flow_module_conf, logger=self.logger)
        self.optimizer = self.configure_optimizers()
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.logger.info(f"Model initialized:\n{self.model}")
        self.epoch_id = 0
        if self.accelerator.is_local_main_process:
            tensorboard_dir = os.path.join(self.exp_conf.tensorboard_dir, self.timestamp)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
            self.ckpt_dir = os.path.join(self.exp_conf.ckpt_dir, self.timestamp)
            os.makedirs(self.ckpt_dir, exist_ok=True)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("Initialization completed successfully")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self.exp_conf.optimizer,
        )

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epoch: int = 1,
    ):
        """
        Train the model using the provided dataset and configurations.
        """
        self.model.train()
        self.step = 0
        self.logger.info(f"Starting training for {num_epoch} epochs")
        self.logger.info(f"Dataset size: {len(train_loader)}")
        self.logger.info(
            f"Max Batch size: {self.exp_conf.dataset_conf.na_conf.sampler_conf.max_batch_size}"
        )
        train_epoch = num_epoch - self.epoch_id
        for epoch in range(train_epoch):
            self.epoch_id += 1
            self.logger.info(f"Epoch {self.epoch_id}/{num_epoch}")
            epoch_losses = {
                "train_loss": [],
                "trans_loss": [],
                "rots_vf_loss": [],
                "aatypes_loss": [],
                "torsion_loss": [],
            }
            for i, batch in enumerate(train_loader):
                with self.accelerator.accumulate(self.model):
                    self.step += 1
                    if self.device.type == "cuda":
                        batch = cuda_utils.cuda(batch)
                    total_loss = self.model(batch)
                    train_loss = total_loss["train_loss"]
                    if train_loss is None:
                        self.save()
                        raise ValueError("Training NAN")
                    self.accelerator.backward(train_loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.exp_conf.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                local_losses = {
                    "train_loss": total_loss["train_loss"].detach(),
                    "trans_loss": total_loss["trans_loss"].detach(),
                    "rots_vf_loss": total_loss["rots_vf_loss"].detach(),
                    "aatypes_loss": total_loss["aatypes_loss"].detach(),
                    "torsion_loss": total_loss["torsion_loss"].detach(),
                }

                for loss_name, loss_value in local_losses.items():
                    epoch_losses[loss_name].append(loss_value)

                if (i + 1) % 10 == 0:
                    if self.accelerator.is_local_main_process:
                        for loss_name, loss_value in local_losses.items():
                            self.writer.add_scalar(f"Train/{loss_name}", loss_value, self.step)
                        loss_msg = " ".join(
                            [f"{name}: {value:.4f}" for name, value in local_losses.items()]
                        )
                        self.logger.info(
                            f"Epoch: {self.epoch_id}/{num_epoch} Batch: {i + 1}/{len(train_loader)} | {loss_msg}"
                        )

            if self.accelerator.is_local_main_process:
                for loss_name, avg_loss in epoch_losses.items():
                    self.logger.info(f"Average {loss_name}: {sum(avg_loss) / len(avg_loss):.4f}")
                    self.writer.add_scalar(
                        f"Epoch/{loss_name}", sum(avg_loss) / len(avg_loss), self.epoch_id
                    )
                self.logger.info(f"Epoch {self.epoch_id} summary completed.")

            if (epoch + 1) % 1 == 0:
                self.save()

    def validation(
        self,
        valid_dataloader: torch.utils.data.DataLoader,
    ):
        self.model.eval()
        self.logger.info("Starting test")
        self.logger.info(f"Dataset size: {len(valid_dataloader)}")

        self.logger.info(
            f"Max Batch size: {self.exp_conf.dataset_conf.sampler_conf.max_batch_size}"
        )

        validation_dir = os.path.join(
            self.exp_conf.validation_dir,
            self.timestamp,
            f"epcoh_{self.epoch_id}",
        )

        metrics_path = os.path.join(
            validation_dir,
            "validation_statistics.csv",
        )

        os.makedirs(validation_dir, exist_ok=True)
        validation_epoch_metrics = []
        for batch_idx, batch in enumerate(valid_dataloader):
            if self.device.type == "cuda":
                batch = cuda_utils.cuda(batch)
            batch_metrics = self.model.validation_step(batch, batch_idx, validation_dir)
            validation_epoch_metrics.append(batch_metrics)
            if batch_idx % 10 == 0:
                self.logger.info(f"Processed {batch_idx} batches")
        val_epoch_metrics = pd.concat(validation_epoch_metrics, ignore_index=True)
        val_epoch_metrics.to_csv(metrics_path, index=True)

    def evaluation(
        self,
        eval_dataloader: torch.utils.data.DataLoader,
        lig_feats: dict | None = None,
    ):
        self.model.eval()
        self.logger.info("Starting evaluation")
        self.logger.info(f"Dataset size: {len(eval_dataloader)}")

        evaluation_dir = os.path.join(
            self.exp_conf.evaluation_dir,
            self.timestamp,
        )

        evaluation_metrics_path = os.path.join(
            evaluation_dir,
            "eval_statistics.csv",
        )

        if lig_feats is not None and self.device.type == "cuda":
            lig_feats = cuda_utils.cuda(lig_feats)

        os.makedirs(evaluation_dir, exist_ok=True)
        evaluation_epoch_metrics = []
        for batch_idx, batch in enumerate(eval_dataloader):
            if self.device.type == "cuda":
                batch = cuda_utils.cuda(batch)
            batch_metrics = self.model.eval_step(batch, evaluation_dir, lig_feats)
            evaluation_epoch_metrics.append(batch_metrics)
        val_epoch_metrics = pd.concat(evaluation_epoch_metrics, ignore_index=True)
        val_epoch_metrics.to_csv(evaluation_metrics_path, index=True)

    def create_tensorboard(self):
        if self.accelerator.is_local_main_process:
            tensorboard_dir = os.path.join(self.exp_conf.tensorboard_dir, self.timestamp)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)

    def save(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.logger.warning("Save checkpoint to %s" % self.ckpt_dir)
            ckpt_path = os.path.join(self.ckpt_dir, f"{self.epoch_id}.ckpt")
            state = {
                "model": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch_id": self.epoch_id,
            }
            torch.save(state, ckpt_path)
            self.logger.info("Checkpoint saved successfully")

    def load(self, checkpoint_path, load_optimizer=True):
        # fmt: off
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model_state = checkpoint["model"]
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            missing_keys, unexpected_keys = unwrapped_model.load_state_dict(model_state, strict=False)
            self.model = self.accelerator.prepare(unwrapped_model)
            if missing_keys or unexpected_keys:
                self.logger.warning(f"Missing keys: {missing_keys}")
                self.logger.warning(f"Unexpected keys: {unexpected_keys}")
            if load_optimizer:
                optimizer_state = checkpoint["optimizer"]
                for state in optimizer_state["state"].values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.accelerator.device)
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.epoch_id = checkpoint["epoch_id"]
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}.")
        else:
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist.")

    def setup_logger(self, exp_conf: Any) -> logging.Logger:
        """
        Setup logger using Accelerate's built-in functionality for rank=0.
        """
        # fmt: off
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO if self.accelerator.is_local_main_process else logging.WARN)

        # Rank 0 process logs to file
        if self.accelerator.is_local_main_process:
            log_dir = exp_conf.log_dir
            log_filename = os.path.join(log_dir, f"riboflow_{self.timestamp}.log")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_filename}")
            logger.info("Initializing RiboFlow Engine with configurations:")
            logger.info("-" * 50)
            logger.info(json.dumps(OmegaConf.to_container(exp_conf, resolve=True), indent=2))
            logger.info("-" * 50)
        return logger
