import sys
sys.path.append('../') # ensure correct path dependency

import argparse
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.schema import Schema
from trainer import Trainer
import wandb
from root import root   
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dirpath", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=False, default=0)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    cs = ConfigStore.instance()
    cs.store(name="schema", node=Schema)
    initialize(version_base=None, config_path=args.config_dirpath)
    dict_cfg: DictConfig = compose(config_name=args.config_name)

    cfg: Schema = OmegaConf.to_object(dict_cfg)
    cfg.subset_size = args.subset_size

    # Print important configs
    print("=" * 50)
    print("Summary of Configuration:")
    print("=" * 50)
    print(f"pe_method: {cfg.pe_method}")
    print(f"  basis: {cfg.basis}")
    print(f"  num_samples: {cfg.num_samples}")
    print(f"  pearl_k: {cfg.pearl_k}")
    print(f"  pe_dims: {cfg.pe_dims}")
    print(f"sample_aggr_model_name: {cfg.sample_aggr_model_name}")
    print(f"  n_sample_aggr_layers: {cfg.n_sample_aggr_layers}")
    print(f"  sample_aggr_hidden_dims: {cfg.sample_aggr_hidden_dims}")
    print(f"  dial_diag_est_place: {cfg.dial_diag_est_place}")
    print(f"base_model: {cfg.base_model}")
    print(f"  n_base_layers: {cfg.n_base_layers}")
    print(f"  base_hidden_dims: {cfg.base_hidden_dims}")
    print("=" * 50)

    if cfg.wandb:
        if not args.testing:
            wandb.login(key="")  # use your own WandB key
        wandb.init(
            mode="disabled" if args.testing else "online",
            dir=root("."), 
            project=cfg.wandb_project, 
            name=cfg.wandb_run_name, 
            config=cfg.__dict__
        )
    
    seeds = [0,1,2,3,4]
    test_rocaucs = []
    for seed in seeds:
        cfg.seed = seed
        trainer = Trainer(cfg, args.gpu_id)
        best_test_perf = trainer.train()
        test_rocaucs.append(best_test_perf)
        print(f"For seed {seed}, test ROCAUC: {best_test_perf}")
    
    mean = np.mean(test_rocaucs)
    std = np.std(test_rocaucs)
    wandb.run.summary["Test(ROCAUC) Mean"] = mean
    wandb.run.summary["Test(ROCAUC) Std"] = std


if __name__ == "__main__":
    main()