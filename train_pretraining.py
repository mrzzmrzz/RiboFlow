from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from src.data.dataset.loader import collate_fn_NA
from src.data.dataset.na_complex_dataset import NABaseDataset
from src.data.dataset.sampler import NABatchSampler
from src.model.engine import Engine


if __name__ == "__main__":
    exp_conf = OmegaConf.load("config/pretrain_se3_cond.yaml")
    na_dataset_conf = exp_conf.dataset_conf.na_conf
    lig_dataset_conf = exp_conf.dataset_conf.ligand_conf
    na_dataset = NABaseDataset(na_dataset_conf)

    metadata_csv = na_dataset.csv

    batch_sampler = NABatchSampler(
        sampler_conf=exp_conf.dataset_conf.na_conf.sampler_conf,
        metadata_csv=metadata_csv,
    )

    dataloader = DataLoader(
        dataset=na_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn_NA,
    )

    trainer = Engine(exp_conf)
    trainer.train(dataloader, exp_conf.exp_conf.num_epoch)
