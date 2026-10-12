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

    if cfg.wandb:
        if not args.testing:
            wandb.login(key="")  # use your own WandB key
        wandb.init(
            mode="disabled" if args.testing else "online",
            dir=root("."), 
            project="ICML-2026", 
            name=cfg.wandb_run_name, 
            config=cfg.__dict__,
            settings=wandb.Settings(code_dir=".")
        )


    tasks = ["C2","C3","C4"]
    seeds = [0] 
    test_losses = []
    for s in seeds:
        cfg.seed = s
        trainer = Trainer(cfg, args.gpu_id)
        best_test_loss = trainer.train()
        test_losses.append(best_test_loss)

    all_losses = torch.stack(test_losses, dim=0)/8192.0        
    mean_loss = all_losses.mean(dim=0)              
    std_loss  = all_losses.std(dim=0, unbiased=False)

    print(f"\nMean test loss: {mean_loss.mean().item():.2f}, std test loss: {std_loss.mean().item():.2f}")
    row = " | ".join(f"{n}: {m:.2f}±{s:.2f}" for n, m, s in zip(tasks, mean_loss.tolist(), std_loss.tolist()))
    print("Per-task (mean±std over 5 runs):", row)

    table = wandb.Table(columns=["Task", "Mean Loss", "Std Loss"])
    for task, m, s in zip(tasks, mean_loss.tolist(), std_loss.tolist()):
        table.add_data(task, m, s)
    wandb.log({"per_task_loss": table})


if __name__ == "__main__":
    main()