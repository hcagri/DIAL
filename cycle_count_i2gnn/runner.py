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
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument(
        "--out_dirpath",
        type=str,
        default=None,
        help="Optional output directory overriding the value in the config.",
    )
    parser.add_argument(
        "--cycles", type=str, default="3,4,5,6",
        help="Comma-separated cycle lengths to run (subset of 3,4,5,6).",
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
            project="ICML-2026", 
            name=cfg.wandb_run_name, 
            config=cfg.__dict__,
            settings=wandb.Settings(code_dir=".")
        )


    # Configuration
    cycles = [
        int(cycle.strip())
        for cycle in args.cycles.split(',')
        if cycle.strip()
    ]
    if not cycles:
        raise ValueError("--cycles must contain at least one cycle length")
    invalid_cycles = sorted(set(cycles) - {3, 4, 5, 6})
    if invalid_cycles:
        raise ValueError(f"Unsupported cycle lengths: {invalid_cycles}")
    seeds = range(args.seed, args.seed + args.num_seeds)
    
    # Collect results: {cycle: [loss_per_seed]}
    results = {cycle: [] for cycle in cycles}
    results_normed = {cycle: [] for cycle in cycles}
    
    # Nested loop: for each cycle-task, run training across all seeds
    for cycle in cycles:
        cfg.task = cycle - 3
        for seed in seeds:
            cfg.seed = seed
            trainer = Trainer(cfg, cycle-3, args.gpu_id)
            best_test_loss, best_test_loss_normed = trainer.train(cfg.seed)
            results[cycle].append(best_test_loss)
            results_normed[cycle].append(best_test_loss_normed)
    
    # Aggregate results per cycle
    stats = {}
    stats_normed = {}
    for cycle in cycles:
        losses = torch.tensor(results[cycle])
        mean = losses.mean().item()
        std = losses.std(unbiased=False).item() if len(seeds) > 1 else 0.0
        stats[cycle] = {"mean": mean, "std": std}
        losses_normed = torch.tensor(results_normed[cycle])
        mean_normed = losses_normed.mean().item()
        std_normed = losses_normed.std(unbiased=False).item() if len(seeds) > 1 else 0.0
        stats_normed[cycle] = {"mean": mean_normed, "std": std_normed}
    
    # Print formatted table
    col_width = 14
    header = " | ".join(f"{c}-cycle".center(col_width) for c in cycles)
    separator = "-+-".join("-" * col_width for _ in cycles)
    values = " | ".join(
        f"{stats[c]['mean']:.2f} ± {stats[c]['std']:.2f}".center(col_width)
        for c in cycles
    )
    values_normed = " | ".join(
        f"{stats_normed[c]['mean']:.2f} ± {stats_normed[c]['std']:.2f}".center(col_width)
        for c in cycles
    )
    
    print("\n" + "=" * len(header))
    print(header)
    print(separator)
    print(values)
    print(values_normed)
    print("=" * len(header))
    
    # Log mean results to wandb
    if cfg.wandb:
        for cycle in cycles:
            wandb.log({f"best_test_loss_cycle_{cycle}": stats[cycle]["mean"]})
            wandb.log({f"best_test_loss_normed_cycle_{cycle}": stats_normed[cycle]["mean"]})


if __name__ == "__main__":
    main()
