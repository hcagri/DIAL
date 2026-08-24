import sys
sys.path.append('../') # ensure correct path dependency

import argparse
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.schema import Schema
from trainer import Trainer
import torch
import wandb
from root import root


def main():
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

    if cfg.wandb:
        wandb.login(key="") # use your own Wandb key
        wandb.init(
            mode="disabled" if args.testing else "online",
            dir=root("."), 
            project="ICML-2026", 
            name=cfg.wandb_run_name, 
            config=cfg.__dict__,
            settings=wandb.Settings(code_dir=".")
        )


    # tasks = ["deg-in","deg-out","fan-in","fan-out","C2","C3","C4","C5","C6","S-G","B-C"]
    tasks = ["deg-in","deg-out","C2","C3","C4","C5","C6","S-G","B-C"]
    seeds = [0,1,2,3,4]
    test_f1_scores = []
    for s in seeds:
        cfg.seed = s
        trainer = Trainer(cfg, args.gpu_id)
        best_test_f1 = trainer.train()
        test_f1_scores.append(best_test_f1.cpu())

    all_f1 = torch.stack(test_f1_scores, dim=0)        
    mean_f1 = all_f1.mean(dim=0)              
    std_f1  = all_f1.std(dim=0, unbiased=False)

    macro_mean = mean_f1.mean().item()*100

    print(f"\nMacro minority F1 over 5 runs: {macro_mean:.2f}%")
    row = " | ".join(f"{n}: {100*m:.2f}±{100*s:.2f}%" for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()))
    print("Per-task (mean±std over 5 runs):", row)

    table = wandb.Table(columns=["Task", "Mean F1 (%)", "Std F1 (%)"])
    for task, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()):
        table.add_data(task, m * 100, s * 100)
    wandb.log({"per_task_f1": table})



if __name__ == "__main__":
    main()