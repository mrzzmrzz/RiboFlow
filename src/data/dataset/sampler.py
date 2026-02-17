import logging
import math
from copy import deepcopy

import torch
from torch.utils.data import BatchSampler

from src.deps.exper import comms


class NABatchSampler(BatchSampler):
    def __init__(
        self,
        sampler_conf,
        metadata_csv,
        batch_size=1,
        drop_last=False,
        seed=13,
        shuffle=True,
        num_replicas=None,
        rank=None,
    ):
        """
        A custom batch sampler for distributed data loading.
        """
        super().__init__(None, batch_size=batch_size, drop_last=drop_last)  # type: ignore

        self.logger = logging.getLogger(__name__)

        # Distributed setup
        self.num_replicas = num_replicas or comms.get_world_size()
        self.rank = rank or comms.get_rank()
        print("envs:", self.rank)

        self.sampler_conf = sampler_conf
        self.metadata_csv = deepcopy(metadata_csv)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        # Total and per-replica batch calculations
        self.overall_num_batches = len(self.metadata_csv)
        self.num_batches = math.ceil(self.overall_num_batches / self.num_replicas)

        # Add an index column for sample tracking
        self.metadata_csv["index"] = list(range(len(self.metadata_csv)))

        # Maximum batch size
        self.max_batch_size = self.sampler_conf.max_batch_size

        self.logger.info(f"Created dataloader rank {self.rank + 1} of {self.num_replicas}")

    def _replica_epoch_batches(self):
        """
        Generate batches for the current replica in this epoch.
        """
        # Set random seed for reproducibility
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        # Shuffle or use sequential order
        indices = (
            torch.randperm(len(self.metadata_csv), generator=rng).tolist()
            if self.shuffle
            else list(range(len(self.metadata_csv)))
        )

        # Filter data for the current replica
        if len(self.metadata_csv) > self.num_replicas:
            # Each replica_csv is a subset of the entire csv, evenly divided.
            replica_csv = self.metadata_csv.iloc[indices[self.rank :: self.num_replicas]]
        else:
            replica_csv = self.metadata_csv

        # Group by sequence length and create batches
        sample_order = []
        for seq_len, len_df in replica_csv.groupby("modeled_na_seq_len"):
            max_batch_size = min(
                self.max_batch_size,
                self.sampler_conf.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i * max_batch_size : (i + 1) * max_batch_size]
                batch_indices = batch_df["index"].tolist()
                sample_order.append(batch_indices)

        # Shuffle batches to remove length bias
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).tolist()
            sample_order = [sample_order[i] for i in new_order]

        return sample_order

    def __iter__(self):
        """
        Return an iterator for the batches in the current epoch.
        """
        self.sample_order = []
        num_augments = -1
        # total training sample number of one-epoch: batch_size * self.num_batches
        # Ensure consistent number of batches across replicas
        while len(self.sample_order) < self.num_batches:
            self.sample_order.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError("Exceeded number of augmentations.")

        self.sample_order = self.sample_order[: self.num_batches]
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        """
        Return the number of batches for this sampler.
        """
        if hasattr(self, "sample_order"):
            return len(self.sample_order)
        else:
            return self.num_batches
