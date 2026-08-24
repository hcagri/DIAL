#!/usr/bin/env bash
# Smoke-test Hydra configs: load config, build model, run one train step on CPU.
# Usage (from PEARL root or anywhere):
#   ./test_configs.sh
#   ./test_configs.sh configs/pattern
#   GPU_ID=0 ./test_configs.sh configs/zinc   # optional GPU (default: CPU)

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ROOT="${1:-$ROOT/configs}"
# Empty / unset => CPU. Set GPU_ID=0 (etc.) to use a GPU.
GPU_ID="${GPU_ID:-}"

if [[ ! -d "$CONFIG_ROOT" ]]; then
  echo "Config directory not found: $CONFIG_ROOT"
  exit 1
fi

if [[ -z "$GPU_ID" ]]; then
  echo "Using CPU for smoke tests"
else
  echo "Using GPU $GPU_ID for smoke tests"
fi

map_experiment() {
  local rel="$1"
  case "$rel" in
    configs/pattern/*|pattern/*) echo "pattern" ;;
    configs/zinc/*|zinc/*) echo "zinc" ;;
    configs/cycle_counts_i2gnn/*|cycle_counts_i2gnn/*) echo "cycle_count_i2gnn" ;;
    configs/cycle_counts/*|cycle_counts/*) echo "cycle_count" ;;
    configs/molhiv/*|molhiv/*) echo "molhiv" ;;
    configs/peptides/*|peptides/*) echo "peptides" ;;
    configs/drugood/*|drugood/*) echo "drugood" ;;
    *) echo "" ;;
  esac
}

PASS=0
FAIL=0
SKIP=0
FAILED_LIST=()

while IFS= read -r -d '' yaml; do
  rel="${yaml#"$ROOT"/}"
  name="$(basename "$yaml" .yaml)"
  dir="$(dirname "$yaml")"
  exp="$(map_experiment "$rel")"

  if [[ -z "$exp" ]]; then
    echo "SKIP  $rel  (no runner mapping)"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo "============================================================"
  echo "TEST  $rel  ->  $exp / $name"
  echo "============================================================"

  if python - "$ROOT" "$dir" "$name" "$exp" "$GPU_ID" <<'PY'
import os
import sys

root, config_dir, config_name, experiment, gpu_id_str = sys.argv[1:]
gpu_id = None if gpu_id_str == "" else int(gpu_id_str)

sys.path.insert(0, root)
os.chdir({
    "pattern": os.path.join(root, "patterns"),
    "zinc": os.path.join(root, "zinc"),
    "cycle_count_i2gnn": os.path.join(root, "cycle_count_i2gnn"),
    "cycle_count": os.path.join(root, "cycle_count"),
    "molhiv": os.path.join(root, "molhiv"),
    "peptides": os.path.join(root, "peptides"),
    "drugood": os.path.join(root, "drugood"),
}[experiment])
sys.path.insert(0, os.getcwd())

from hydra import initialize_config_dir, compose
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from src.schema import Schema
from trainer import Trainer

GlobalHydra.instance().clear()
cs = ConfigStore.instance()
cs.store(name="schema", node=Schema)
initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir))
cfg = OmegaConf.to_object(compose(config_name=config_name))
cfg.wandb = False
cfg.n_epochs = 1
cfg.seed = 0
# Keep smoke tests light on memory (especially pattern / DIAL).
if hasattr(cfg, "train_batch_size") and cfg.train_batch_size is not None:
    cfg.train_batch_size = min(int(cfg.train_batch_size), 8)
if hasattr(cfg, "num_samples") and cfg.num_samples is not None:
    cfg.num_samples = min(int(cfg.num_samples), 8)

if experiment == "cycle_count_i2gnn":
    trainer = Trainer(cfg, target=0, gpu_id=gpu_id)
else:
    trainer = Trainer(cfg, gpu_id)

batch = next(iter(trainer.train_loader))
trainer.model.train()
loss = trainer.train_batch(batch)
print(f"OK  one train step on {trainer.device}, loss={float(loss):.6g}")
PY
  then
    echo "PASS  $rel"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $rel"
    FAIL=$((FAIL + 1))
    FAILED_LIST+=("$rel")
  fi
  echo
done < <(find "$CONFIG_ROOT" -name '*.yaml' -print0 | sort -z)

echo "============================================================"
echo "PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
  echo "Failed configs:"
  printf '  %s\n' "${FAILED_LIST[@]}"
  exit 1
fi
exit 0
