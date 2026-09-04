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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dirpath", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=False, default=0)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=4)
    parser.add_argument(
        "--out_dirpath",
        type=str,
        default=None,
        help="Optional output directory overriding the value in the config.",
    )
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be at least 1")

    cs = ConfigStore.instance()
    cs.store(name="schema", node=Schema)
    initialize(version_base=None, config_path=args.config_dirpath)
    dict_cfg: DictConfig = compose(config_name=args.config_name)

    cfg: Schema = OmegaConf.to_object(dict_cfg)
    cfg.subset_size = args.subset_size
    if args.out_dirpath is not None:
        cfg.out_dirpath = args.out_dirpath

    if cfg.wandb:
        wandb.init(
            mode="disabled" if args.testing else "online",
            dir=root("."), 
            project=cfg.wandb_project, 
            name=cfg.wandb_run_name, 
            config=cfg.__dict__
        )
    total_test_loss = 0
    for i in range(args.num_seeds):
        cfg.seed = args.seed + i
        trainer = Trainer(cfg, args.gpu_id)
        test_loss_best = trainer.train(cfg.seed)
        print(f"Test loss best on seed {cfg.seed}: {test_loss_best}")
        total_test_loss += test_loss_best
        
    if cfg.wandb:
        wandb.run.summary["ave_test_loss"] = total_test_loss / args.num_seeds


if __name__ == "__main__":
    main()
