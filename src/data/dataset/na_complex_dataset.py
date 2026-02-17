from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.basics import utils as du


def init_metadata_and_splits(data_conf: Any) -> pd.DataFrame:
    pdb_csv = pd.read_csv(data_conf.csv_path)

    # original CSV before filtering and transformations
    max_len: int = data_conf.filtering.max_len
    min_len: int = data_conf.filtering.min_len

    # length-based filtering: modeled_na_seq_len
    pdb_csv = pdb_csv[pdb_csv["modeled_na_seq_len"] <= max_len]
    pdb_csv = pdb_csv[pdb_csv["modeled_na_seq_len"] >= min_len]

    pdb_csv = pdb_csv.sort_values(by=["modeled_na_seq_len"], ascending=False)  # type:ignore

    if data_conf.is_training:
        csv = pdb_csv
        print(f"Training: {len(csv)} filtered examples")
    else:
        eval_csv = pdb_csv
        all_lengths = pdb_csv["modeled_na_seq_len"].unique()
        length_indices = (len(all_lengths) - 1) * np.linspace(
            0.0, 1.0, data_conf.num_eval_lengths
        )

        length_indices = length_indices.astype(int)
        eval_lengths = all_lengths[length_indices]
        eval_csv = eval_csv[eval_csv["modeled_na_seq_len"].isin(eval_lengths)]  # type:ignore

        eval_csv = eval_csv.groupby(by=["modeled_na_seq_len"]).sample(
            data_conf.samples_per_eval_length, replace=True, random_state=42
        )

        eval_csv = eval_csv.sort_values(by=["modeled_na_seq_len"], ascending=False)  # type:ignore

        csv = eval_csv
        print(f"Validation: {len(csv)} examples with lengths {eval_lengths}.")
    return csv


class NABaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_conf,
    ):
        self.data_conf = data_conf
        self.is_training = data_conf.is_training
        self.csv = init_metadata_and_splits(self.data_conf)

    # fmt: on
    def convert_dict_float64_items_to_float32(self, dictionary):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted_dict[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted_dict[key] = value.float()
            else:
                # for non-NumPy array and non-PyTorch tensor types
                converted_dict[key] = value
        return converted_dict

    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row["processed_na_path"]

        # get the features for this instance
        final_feats = du.read_pkl(processed_file_path)
        final_feats = self.convert_dict_float64_items_to_float32(final_feats)

        return final_feats

    def __len__(self):
        return len(self.csv)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


class LigandBaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_conf):
        self.data_conf = data_conf
        self.is_training = data_conf.is_training
        self.csv = init_metadata_and_splits(self.data_conf)

    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row["processed_ligand_path"]

        # get the features for this instance
        final_feats = du.read_pkl(processed_file_path)

        # data = Data(
        #     x=final_feats["x"].type(torch.float32),
        #     edge_index=torch.tensor(final_feats["edge_index"], dtype=torch.long),
        #     edge_attr=final_feats["edge_features"].type(torch.float32),
        # )

        feats = {
            "ligand_feat": final_feats["ligand_feat"],
            "ligand_mask": final_feats["ligand_mask"],
            "ligand_pos": final_feats["ligand_pos_after_com"],
        }

        return feats

    def __len__(self):
        return len(self.csv)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


class NAComplexBaseDataset(torch.utils.data.Dataset):
    def __init__(self, na_dataset, ligand_dataset):
        self.na_dataset = na_dataset
        self.ligand_dataset = ligand_dataset

    def __getitem__(self, idx):
        na_data = self.na_dataset[idx]
        ligand_data = self.ligand_dataset[idx]
        return na_data, ligand_data

    def __len__(self):
        assert len(self.na_dataset) == len(self.ligand_dataset)
        return len(self.na_dataset)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_conf):
        self.samples_conf = samples_conf

        all_sample_lengths = range(
            self.samples_conf.min_len,
            self.samples_conf.max_len + 1,
            self.samples_conf.step_len,
        )

        # ignore the above variable if subset is given
        if self.samples_conf.length_subset is not None:
            all_sample_lengths = [int(x) for x in self.samples_conf.length_subset]

        print(f"#### Generating sequences with the following lengths: {list(all_sample_lengths)}")

        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(self.samples_conf.samples_per_length):
                all_sample_ids.append((length, sample_id))

        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            "num_res": num_res,
            "sample_id": sample_id,
        }
        return batch
