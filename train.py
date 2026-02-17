from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from src.data.dataset.loader import collate_fn
from src.data.dataset.na_complex_dataset import (
    LigandBaseDataset,
    NABaseDataset,
    NAComplexBaseDataset,
)
from src.data.dataset.sampler import NABatchSampler
from src.model.engine import Engine


if __name__ == "__main__":
    exp_conf = OmegaConf.load("config/pretrain_se3_cond.yaml")
    na_dataset_conf = exp_conf.dataset_conf.na_conf
    lig_dataset_conf = exp_conf.dataset_conf.ligand_conf
    na_dataset = NABaseDataset(na_dataset_conf)
    lig_dataset = LigandBaseDataset(lig_dataset_conf)
    train_dataset = NAComplexBaseDataset(na_dataset, lig_dataset)
    metadata_csv = na_dataset.csv
    alist = []
    batch_sampler = NABatchSampler(
        sampler_conf=exp_conf.dataset_conf.na_conf.sampler_conf,
        metadata_csv=metadata_csv,
    )
    for _ in batch_sampler:
        alist.append(_)

    num_item = [len(num) for num in alist]
    print(len(metadata_csv))
    print(sum(num_item))

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )

    trainer = Engine(exp_conf)
    trainer.train(dataloader, exp_conf.exp_conf.num_epoch)
