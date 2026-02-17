import logging
import os
from multiprocessing import Manager, Pool
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import tree
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm

from src.data.basics import data_transforms_na, rigid_utils_na
from src.data.basics import utils as du
from src.data.basics.constants.vocabulary import restype_to_str_sequence
from src.data.parsing import parsers


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_NA_feat(processed_feats: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Process a single processed file and prepare chain features.

    Args:
        processed_file_path (str): Path to the processed feature file.

    Returns:
        dict[str, torch.Tensor]: Final features for a single processed file.
    """
    # processed_feats: dict[Any, Any] = du.read_pkl(processed_feat)
    flow_mask = np.ones_like(processed_feats["bb_mask"])

    if np.sum(flow_mask) < 1:
        raise ValueError("No residues available for diffusion.")

    fixed_mask = 1 - flow_mask
    processed_feats["fixed_mask"] = fixed_mask
    processed_feats["is_na_residue_mask"] = np.any(
        np.array(processed_feats["molecule_type_encoding"])[:, 1:3] == 1, axis=1
    )

    na_modeled_idx = processed_feats["na_modeled_idx"]
    min_idx, max_idx = np.min(na_modeled_idx), np.max(na_modeled_idx)

    processed_feats = tree.map_structure(  # type:ignore
        lambda x: x[min_idx : max_idx + 1]
        if isinstance(x, (np.ndarray, list, torch.Tensor))
        else x,
        processed_feats,
    )

    # def slice_if_array(x):
    #     if isinstance(x, (np.ndarray, list, torch.Tensor)):
    #         return x[min_idx : max_idx + 1]
    #     return x

    # processed_feats = tree.map_structure(slice_if_array, processed_feats)  # type: ignore
    na_inputs_present = processed_feats["is_na_residue_mask"].any().item()

    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        # TODO: centered?
        # "all_atom_positions": torch.tensor(processed_feats["centered_atom_positions"]).double(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),  # ATOM23
        "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
        "atom23_gt_positions": torch.tensor(processed_feats["atom_positions"]).double(),
    }

    if na_inputs_present:
        """
        chain_feats keys:
            - 'aatype'
            - 'all_atom_positions': [num_res, 23, 3]
            - 'all_atom_mask': [num_res, 23]
            - 'atom_deoxy'
            - 'atom23_atom_exists': [num_res, 23]
            - 'residx_atom23_to_atom27': [num_res, 23]
            - 'residx_atom27_to_atom23': [num_res, 27]
            - 'atom27_atom_exists':[num_res, 27]
        """
        na_chain_feats = data_transforms_na.make_atom23_masks(chain_feats)

        """
            - 'all_atom_positions': [num_res, 27, 3]
            - 'all_atom_mask': [num_res, 27]
        """
        data_transforms_na.atom23_list_to_atom27_list(
            na_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
        )

        """
        additional na_chain_feats keys:
          - rigidgroups_gt_frames
          - rigidgroups_gt_exists
          - rigidgroups_group_exists
          - rigidgroups_group_is_ambiguous
          - rigidgroups_alt_gt_frames
        """
        chain_feats = data_transforms_na.atom27_to_frames(na_chain_feats)

        """
        additional na_chain_feats keys:
          - torsion_angles_sin_cos
          - alt_torsion_angles_sin_cos
          - torsion_angles_mask
        """
        chain_feats = data_transforms_na.atom27_to_torsion_angles()(na_chain_feats)  # type: ignore

    chain_feats = du.concat_complex_torch_features(
        chain_feats,
        {},
        na_chain_feats,  # type: ignore
        feature_concat_map=du.COMPLEX_FEATURE_CONCAT_MAP,
        add_batch_dim=False,
    )
    # A C G U -> 0,1,2,3
    processed_feats["aatypes_1"] = chain_feats["aatype"] - 4
    res_idx = torch.arange(chain_feats["aatype"].shape[0])
    rigids_1 = rigid_utils_na.Rigid.from_tensor_4x4(chain_feats["rigidgroups_gt_frames"])[:, 0]
    rotmats_1, trans_1 = rigids_1.get_rots().get_rot_mats(), rigids_1.get_trans()

    final_feats = {
        "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
        "gt_positions": chain_feats["all_atom_positions"],
        "is_na_residue_mask": torch.tensor(processed_feats["is_na_residue_mask"]),
        "aatypes_1": processed_feats["aatypes_1"],
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": torch.tensor(processed_feats["bb_mask"]).type(torch.int),
        "flow_mask": torch.tensor(processed_feats["bb_mask"]).type(torch.int),
        "res_idx": res_idx,
    }
    # print(final_feats)

    return final_feats


def process_NA_file(
    file_path: str,
    write_dir: str,
    skip_existing: bool = False,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """Process a mmCIF file to extract features and metadata.

    Args:
        file_path (str): Path to the PDB or CIF file.
        write_dir (str): Directory to save processed data.
        skip_existing (bool): Skip processing if output file already exists.
        verbose (bool): Verbose output for debugging.

    Returns:
        Optional[dict[str, Any]]: Metadata dictionary or None if processing fails.
    """
    metadata = {}
    pdb_name, format = (
        os.path.splitext(os.path.basename(file_path))[0],
        os.path.splitext(file_path)[1][1:],
    )
    metadata["pdb_name"] = pdb_name

    pdb_subdir = write_dir
    processed_path = os.path.join(pdb_subdir, f"{pdb_name}.pkl")
    metadata["processed_na_path"] = processed_path

    if skip_existing and os.path.exists(processed_path):
        return None

    if format == "pdb":
        parser = PDBParser(QUIET=True)
    elif format == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        logger.error(f"Unsupported data format: {format}")
        return None

    try:
        structure = parser.get_structure(pdb_name, file_path)
        if structure is None:
            logger.warning(f"{pdb_name} structure not found. Please check the source file.")
            return None
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None

    struct_chains = {chain.id: chain for chain in structure.get_chains()}
    struct_feats = []
    num_na_chains = 0
    na_natype = torch.empty(0)

    for chain_id, chain in struct_chains.items():
        chain_index = du.chain_str_to_int(chain_id)

        # chain_mol:
        #   - X: position
        #   - C: chain
        #   - S: sequence
        #   - metadata: dict
        #       - atom_mask
        #       - b_factors
        #       - chain_ids (alphabet)
        #       - deoxy
        #       - source_file (optional)
        #       - is_valid_molecule_type (True, False)
        #       - molecule_type (protein, na)
        #       - molecule_type_encoding
        #           - (1, 0, 0, 0) protein
        #           - (0, 1, 0, 0) DNA
        #           - (0, 0, 1, 0) RNA
        #       - molecule_constants
        #           - protein_constant.py
        #           - nucleotide_constant.py
        #       - molecule_backbone_atom_name
        #           - "CA" (protein)
        #           - "C4" (na)

        chain_mol = parsers.process_chain_pdb(chain, chain_index, chain_id, verbose=verbose)

        if chain_mol is None:
            continue

        if chain_mol[-1]["molecule_type"] == "na":
            num_na_chains += 1
            na_natype = torch.cat((na_natype, chain_mol[-2]), dim=0)

            # chain_dict keys:
            #   - atom_positions (num_res, 23, 3)
            #   - atom_chain_id_mask
            #   - aatype
            #   - atom_mask
            #   - atom_chain_indices
            #   - atom_deoxy
            #   - atom_b_factors
            #   - molecule_type_encoding
            #   - bb_mask
            #   - bb_positions (num_res, 3)
            #   - bb_center (1, 3)
            #   - centered_atom_positions (num_res, 23, 3)

            chain_dict = du.parse_chain_feats_pdb(
                chain_feats=parsers.macromolecule_outputs_to_dict(chain_mol),
                molecule_constants=chain_mol[-1]["molecule_constants"],
                molecule_backbone_atom_name=chain_mol[-1]["molecule_backbone_atom_name"],
            )

            struct_feats.append(chain_dict)

    if not struct_feats:
        if verbose:
            logger.warning(f"No chains found for NA file {file_path}. Skipping...")
        return None

    processed_feats = du.concat_np_features(struct_feats, add_batch_dim=False)

    if np.sum(processed_feats["bb_mask"]) < 1.0:
        if verbose:
            logger.warning(f"No backbone atoms found for NA file {file_path}. Skipping...")
        return None

    metadata["aatype"] = restype_to_str_sequence(processed_feats["aatype"]).upper()
    metadata["na_seq_len"] = na_natype.shape[0]  # type: ignore
    na_modeled_idx = np.where(na_natype != 26)[0]
    processed_feats["na_modeled_idx"] = na_modeled_idx.tolist()
    metadata["modeled_na_seq_len"] = np.max(na_modeled_idx) - np.min(na_modeled_idx) + 1
    metadata["na_bb_center"] = str(processed_feats["bb_center"])

    # extract translation and rotmats information
    processed_feats = parse_NA_feat(processed_feats)
    du.write_pkl(processed_path, processed_feats)
    return metadata


def main_process_NA_file(config) -> None:
    pdb_dir, write_dir, metadata_file_path = (
        config.pdb_dir,
        config.write_dir,
        config.metadata_file_path,
    )
    os.makedirs(write_dir, exist_ok=True)
    all_file_paths = [
        os.path.join(pdb_dir, item)
        for item in os.listdir(pdb_dir)
        if item.endswith((".pdb", ".cif"))
    ]
    total_num_paths = len(all_file_paths)

    logger.info(f"Files will be written to {write_dir}")
    all_metadata = []
    for file_path in tqdm(all_file_paths, desc="Processing files"):
        try:
            metadata = process_NA_file(file_path, write_dir, skip_existing=False, verbose=True)
            if metadata:
                all_metadata.append(metadata)
                logger.info(f"Successfully processed {file_path}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_file_path, index=False)
    logger.info(f"Finished processing {len(all_metadata)}/{total_num_paths} files")


def process_single_file(args: tuple) -> Optional[dict[str, Any]]:
    """Wrapper function for processing a single file in parallel."""
    file_path, write_dir, skip_existing, verbose = args
    try:
        return process_NA_file(file_path, write_dir, skip_existing, verbose)
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None


def main_process_NA_file_parallel(config) -> None:
    """Main function to process NA files in parallel."""
    pdb_dir, write_dir, metadata_file_path = (
        config.pdb_dir,
        config.write_dir,
        config.metadata_file_path,
    )
    os.makedirs(write_dir, exist_ok=True)

    all_file_paths = [
        os.path.join(pdb_dir, item)
        for item in os.listdir(pdb_dir)
        if item.endswith((".pdb", ".cif"))
    ]
    total_num_paths = len(all_file_paths)

    logger.info(f"Files will be written to {write_dir}")
    metadata_list = Manager().list()

    # Arguments for each process
    args_list = [(file_path, write_dir, False, True) for file_path in all_file_paths]

    with Pool(processes=4) as pool:
        with tqdm(total=total_num_paths, desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_single_file, args_list):
                if result:
                    metadata_list.append(result)
                pbar.update(1)

    metadata_df = pd.DataFrame(list(metadata_list))
    metadata_df.to_csv(metadata_file_path, index=False)
    logger.info(f"Finished processing {len(metadata_list)}/{total_num_paths} files")
