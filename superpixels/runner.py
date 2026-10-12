import argparse
import sys
from pathlib import Path

import numpy as np
import wandb
from omegaconf import OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from root import root  # noqa: E402
from superpixels.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a continuous-feature superpixel classification experiment."
    )
    parser.add_argument(
        "--config_dirpath",
        default=str(REPOSITORY_ROOT / "configs" / "superpixels"),
    )
    parser.add_argument("--config_name", default="dial")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument(
        "--out_dirpath",
        default=None,
        help="Optional output directory overriding the value in the config.",
    )
    parser.add_argument("--testing", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf overrides, e.g. num_samples=64 n_epochs=150.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    config_name = args.config_name
    if not config_name.endswith((".yaml", ".yml")):
        config_name += ".yaml"
    config_path = Path(args.config_dirpath).expanduser() / config_name
    cfg = OmegaConf.load(config_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.dataset_name is not None:
        cfg.dataset_name = args.dataset_name
    return cfg


def main() -> None:
    args = parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be at least 1")

    cfg = load_config(args)
    if args.out_dirpath is not None:
        cfg.out_dirpath = args.out_dirpath
    if bool(cfg.wandb):
        wandb.init(
            mode="disabled" if args.testing else "online",
            dir=root("."),
            project=str(cfg.wandb_project),
            name=str(cfg.wandb_run_name),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    test_accuracies = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        cfg.seed = seed
        trainer = Trainer(cfg, None if args.gpu_id < 0 else args.gpu_id)
        best_val_accuracy, test_accuracy = trainer.train()
        test_accuracies.append(test_accuracy)
        print(
            f"Seed {seed}: best validation accuracy={best_val_accuracy:.4f}, "
            f"selected test accuracy={test_accuracy:.4f}"
        )

    mean = float(np.mean(test_accuracies))
    std = float(np.std(test_accuracies))
    print(f"Test accuracy: {mean:.4f} ± {std:.4f} ({args.num_seeds} seeds)")

    if bool(cfg.wandb) and wandb.run is not None:
        wandb.run.summary["test_accuracy_mean"] = mean
        wandb.run.summary["test_accuracy_std"] = std
        wandb.finish()


if __name__ == "__main__":
    main()
