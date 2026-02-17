import numpy as np
import torch
import torch.nn.functional as F


LIGAND_PADDING_FEATS = [
    "ligand_feat",
    "ligand_pos",
    "ligand_mask",
]


MSA_PADDING_FEATS = [
    "msa_1",
    "msa_mask",
    "msa_onehot_1",
    "msa_vectorfield",
    "msa_onehot_0",
    "msa_onehot_t",
    "msa_t",
]


def pad(
    x: np.ndarray | torch.Tensor,
    max_len: int,
    pad_idx: int = 0,
    reverse: bool = False,
    pad_value: float = 0,
) -> np.ndarray | torch.Tensor:
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a NumPy array or PyTorch tensor")
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")

    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)

    if isinstance(x, torch.Tensor):
        # Convert NumPy-style pad_widths to PyTorch-style
        torch_pad_widths = []
        for p in reversed(pad_widths):
            torch_pad_widths.extend(p)
        return F.pad(x, torch_pad_widths, value=pad_value)
    return np.pad(x, pad_widths)


def pad_feats(
    raw_feats,
    max_len,
    padding_feats=None,
):
    if padding_feats is None:
        padding_feats = LIGAND_PADDING_FEATS

    padded_feats = {
        feat_name: pad(feat, max_len)
        for feat_name, feat in raw_feats.items()
        if feat_name in padding_feats
    }

    for feat_name in raw_feats:
        if feat_name not in padding_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats


def ligand_batching(feat_dicts):
    def get_ligand_len(x):
        return x["ligand_mask"].shape[0]

    feat_dicts = [x for x in feat_dicts if x is not None]
    if not feat_dicts:
        raise ValueError("No valid feature dictionaries to batch")
    max_ligand_len = max(get_ligand_len(x) for x in feat_dicts)
    padded_batch = [pad_feats(x, max_ligand_len) for x in feat_dicts]
    return torch.utils.data.default_collate(padded_batch)


def collate_fn(
    batch: list[tuple[dict, dict]],
) -> dict:
    """
    Custom collate_fn function for combining RNA and ligand data into a unified batch.

    Args:
        batch (List[Tuple[Dict[str, torch.Tensor], Batch]]): A list of tuples where each tuple contains:
            - A dictionary representing RNA data, with keys as feature names and values as torch tensors.
            - A PyTorch Geometric Data object representing the ligand data.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: A dictionary containing the combined batch data:
            - "rna": A dictionary where each key corresponds to a feature, and the value is a tensor with combined RNA data.
            - "ligand": A PyTorch Geometric Batch object containing the combined ligand data.
    """
    # Extract RNA and ligand data from the batch
    na_batch_list = [item[0] for item in batch]
    ligand_batch_list = [item[1] for item in batch]

    # Initialize the dictionary to store RNA batch data
    na_batch = {}

    # Get the keys of the first RNA data dictionary
    na_keys = na_batch_list[0].keys()
    for key in na_keys:
        # Use default_collate to combine the values for this key across all RNA dictionaries
        na_batch[key] = torch.utils.data.default_collate([na[key] for na in na_batch_list])

    # Combine the ligand data into a PyTorch Geometric Batch
    ligand_batch = ligand_batching(ligand_batch_list)
    collated_batch = {**na_batch, **ligand_batch}
    return collated_batch


def collate_fn_NA(
    batch: list[dict],
) -> dict:
    """
    Custom collate_fn function for combining RNA and ligand data into a unified batch.

    Args:
        batch (List[Tuple[Dict[str, torch.Tensor], Batch]]): A list of tuples where each tuple contains:
            - A dictionary representing RNA data, with keys as feature names and values as torch tensors.
            - A PyTorch Geometric Data object representing the ligand data.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: A dictionary containing the combined batch data:
            - "rna": A dictionary where each key corresponds to a feature, and the value is a tensor with combined RNA data.
            - "ligand": A PyTorch Geometric Batch object containing the combined ligand data.
    """
    # Extract RNA and ligand data from the batch
    na_batch_list = batch

    # Initialize the dictionary to store RNA batch data
    na_batch = {}

    # Get the keys of the first RNA data dictionary
    na_keys = na_batch_list[0].keys()
    for key in na_keys:
        # Use default_collate to combine the values for this key across all RNA dictionaries
        na_batch[key] = torch.utils.data.default_collate([na[key] for na in na_batch_list])

    collated_batch = {**na_batch}
    return collated_batch
