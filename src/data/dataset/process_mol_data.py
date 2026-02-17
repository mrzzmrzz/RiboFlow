import logging
import os
from typing import Any, Optional

import pandas as pd
import torch
import yaml
from rdkit import Chem
from tqdm import tqdm

from src.data.basics import utils as du
from src.data.basics.molecule.molecule_toolkit import (
    get_atom_features,
    get_edge_features,
    mol_to_geognn_graph_data_raw3d,
)


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_mol_file(
    file_path: str,
    write_dir: str,
    na_bb_center: torch.Tensor | None = None,
    skip_existing: bool = False,
    verbose: bool = False,
) -> dict | None:
    """
    Process a molecular file to extract atomic features and coordinates, considering RNA center coordinates from metadata.

    Parameters
    ----------
    file_path : str
        Path to the reference file for naming
    write_dir : str
        Directory where processed files will be saved
    metadata_df : pd.DataFrame
        DataFrame containing existing metadata with RNA center coordinates
    protein_bb_center : Optional[np.ndarray], optional
        Pre-defined protein backbone center coordinates, by default None
    skip_existing : bool, optional
        If True, skip processing if output file already exists, by default False
    verbose : bool, optional
        If True, print additional processing information, by default False

    Returns
    -------
    Optional[Dict[str, np.ndarray]]
        Dictionary containing processed molecular features:
        - ligand_feat: atomic features array
        - ligand_pos: original atomic coordinates
        - ligand_com: center of mass coordinates
        - ligand_pos_after_com: coordinates after centering
        Returns None if processing fails or file is skipped

    Notes
    -----
    If protein_bb_center is None, the function will try to use RNA center coordinates
    from the metadata_df for the corresponding ligand.
    """

    lig_name = os.path.splitext(os.path.basename(file_path))[0]
    format = file_path.split(".")[1]
    processed_path = os.path.join(write_dir, f"{lig_name}.pkl")
    metadata = {"pdb_name": lig_name, "processed_ligand_path": processed_path}
    if skip_existing and os.path.exists(processed_path):
        return None
    try:
        if format == "sdf":
            mol = Chem.SDMolSupplier(file_path)[0]
            if mol is None:
                mol = Chem.SDMolSupplier(file_path, sanitize=False)[0]
        elif format == "pdb":
            mol = Chem.MolFromPDBFile(file_path)
            if mol is None:
                mol = Chem.MolFromPDBFile(file_path, sanitize=False)
        else:
            raise ValueError("Unsupported Data Format!")
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    metadata["smiles"] = smiles

    conf = mol.GetConformer()
    ligand_atom_feat = []
    ligand_atom_coord = []
    for atom in mol.GetAtoms():  # type: ignore
        positions = conf.GetAtomPosition(atom.GetIdx())
        ligand_atom_feat.append(atom.GetAtomicNum())
        ligand_atom_coord.append([positions.x, positions.y, positions.z])

    ligand_atom_feat = torch.tensor(ligand_atom_feat)
    ligand_atom_coord = torch.tensor(ligand_atom_coord)

    if na_bb_center is not None:
        ligand_bb_center = na_bb_center

    else:
        ligand_bb_center = torch.sum(ligand_atom_coord, dim=0) / (
            torch.sum(ligand_atom_feat.clip(max=1.0)) + 1e-10
        )

    ligand_pos_after_com = ligand_atom_coord - ligand_bb_center
    ligand_atom_mask = torch.ones(ligand_atom_feat.shape[0])
    metadata["num_atom"] = ligand_atom_feat.shape[0]
    processed_feats = {
        "ligand_feat": ligand_atom_feat,
        "ligand_mask": ligand_atom_mask,
        "ligand_pos": ligand_atom_coord,
        "ligand_com": ligand_bb_center,  # com: center of mass
        "ligand_pos_after_com": ligand_pos_after_com,
    }
    du.write_pkl(processed_path, processed_feats)

    return metadata


def main_process_mol_file(config) -> None:
    pdb_dir, write_dir, metadata_file_path, metadata_na_file_path = (
        config.pdb_dir,
        config.write_dir,
        config.metadata_file_path,
        config.metadata_na_file_path,
    )
    os.makedirs(write_dir, exist_ok=True)
    all_file_paths = [
        os.path.join(pdb_dir, item) for item in os.listdir(pdb_dir) if item.endswith((".pdb"))
    ]

    total_num_paths = len(all_file_paths)
    logger.info(f"Files will be written to {write_dir}")

    na_metadata_df = pd.read_csv(metadata_na_file_path)

    all_metadata = []
    for file_path in tqdm(all_file_paths, desc="Processing files"):
        try:
            lig_name = os.path.splitext(os.path.basename(file_path))[0]
            idx = na_metadata_df.index[na_metadata_df["pdb_name"] == lig_name]
            na_bb_center = (
                None
                if idx.empty
                else torch.tensor(eval(na_metadata_df.loc[idx, "na_bb_center"].item()))
            )

            # NOTE
            na_bb_center = None

            metadata = process_mol_file(
                file_path,
                write_dir,
                na_bb_center=na_bb_center,
                skip_existing=False,
                verbose=True,
            )
            all_metadata.append(metadata)
            logger.info(f"Successfully processed {file_path}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_file_path, index=False)
    logger.info(f"Finished processing {len(all_metadata)}/{total_num_paths} files")


def process_ligand_file(
    file_path: str,
    write_dir: str,
    skip_existing: bool = False,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Process a ligand SDF file to extract molecular features and save them in a pickle format.

    Parameters
    ----------
    file_path : str
        Path to the input ligand SDF file
    write_dir : str
        Directory where processed files will be saved
    skip_existing : bool, optional
        If True, skip processing if output file already exists, by default False
    verbose : bool, optional
        If True, print additional processing information, by default False

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing metadata about the processed ligand, including:
        - pdb_name: name of the ligand
        - processed_ligand_path: path to the processed pickle file
        - ligand_atom_num: number of atoms in the ligand
        - ligand_edge_num: number of edges in the molecular graph
        Returns None if processing fails or file is skipped

    Notes
    -----
    The processed features saved in the pickle file include:
    - x: atom features (torch.Tensor)
    - edge_index: edge connectivity (torch.Tensor)
    - edge_features: edge features (torch.Tensor)
    """
    metadata = {}
    lig_name = os.path.splitext(os.path.basename(file_path))[0]
    metadata["pdb_name"] = lig_name

    pdb_subdir = write_dir
    processed_path = os.path.join(pdb_subdir, f"{lig_name}.pkl")
    metadata["processed_ligand_path"] = processed_path

    if skip_existing and os.path.exists(processed_path):
        return None

    try:
        mol = Chem.SDMolSupplier(file_path, sanitize=False)[0]
        smiles = Chem.MolToSmiles(mol)
        feat_dict = mol_to_geognn_graph_data_raw3d(mol)
        ligand_pos = feat_dict["atom_pos"]
        edge_index = feat_dict["edges"]
        atom_features = get_atom_features(feat_dict)
        edge_features = get_edge_features(feat_dict)

    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None

    metadata["ligand_atom_num"] = feat_dict["atom_pos"].shape[0]
    metadata["ligand_edge_num"] = edge_index.shape[0]

    processed_feats = {
        "ligand_feat": atom_features,
        "ligand_pos": ligand_pos,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "smiles": smiles,
    }
    du.write_pkl(processed_path, processed_feats)

    return metadata


def main_process_ligand_file(config) -> None:
    pdb_dir, write_dir, metadata_file_path = (
        config.pdb_dir,
        config.write_dir,
        config.metadata_file_path,
    )
    os.makedirs(write_dir, exist_ok=True)
    all_file_paths = [
        os.path.join(pdb_dir, item) for item in os.listdir(pdb_dir) if item.endswith((".sdf"))
    ]
    total_num_paths = len(all_file_paths)

    logger.info(f"Files will be written to {write_dir}")
    all_metadata = []
    for file_path in tqdm(all_file_paths, desc="Processing files"):
        try:
            metadata = process_ligand_file(
                file_path, write_dir, skip_existing=False, verbose=True
            )
            if metadata:
                all_metadata.append(metadata)
                logger.info(f"Successfully processed {file_path}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_file_path, index=False)
    logger.info(f"Finished processing {len(all_metadata)}/{total_num_paths} files")
