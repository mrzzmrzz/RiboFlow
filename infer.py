import os

import torch
from omegaconf import OmegaConf

from src.data.dataset.na_complex_dataset import LengthDataset
from src.model.engine import Engine
from src.model.eval.eval_structure import EvalSuite


epoch_id = 42


def sampling(exp_conf):
    eval_dataset = LengthDataset(exp_conf.evalsuite_conf.eval_conf)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    ckpt_timestep = "xxxxxxxxxx"
    trainer = Engine(exp_conf)
    checkpoint_path = f"ckpt/{ckpt_timestep}/{epoch_id}.ckpt"
    trainer.load(checkpoint_path)
    trainer.evaluation(eval_dataloader)


def eval_sampling(exp_conf):
    sampling_timestep = "xxxxxxxxxx"
    save_dir = f"eval_suite/{sampling_timestep}"
    evalsuite = EvalSuite(
        save_dir,
        paths=exp_conf.evalsuite_conf.paths,
        constants=exp_conf.evalsuite_conf.constants,
        gpu_id1=1,
        gpu_id2=1,
    )

    evalsuite.perform_eval(
        gen_dir=f"evaluation_dir/{sampling_timestep}",
        source="eval_suite",
    )
    metrics_fp = os.path.join(save_dir, "final_metrics.pt")
    metric_dict = evalsuite.load_from_metric_dict(metrics_fp)
    evalsuite.print_global_metrics(metric_dict)  # print eval metrics
    evalsuite.print_local_metrics(metric_dict)  # print eval metrics
