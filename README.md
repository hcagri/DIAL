# Random Probing for Structural Self-Interactions in Graph Neural Networks

Official code for **"Random Probing for Structural Self-Interactions in Graph Neural Networks"**, published at the Learning on Graphs Conference (LoG 2026).

## Installation

Use Python 3.9 and a CUDA build of PyTorch that matches your system. The code was tested with PyTorch 2.2 (CUDA 11.8) and PyTorch Geometric 2.5.2.

```bash
conda create -n dial python=3.9
conda activate dial
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r req_pyg.txt
```

## Data

| Experiment | Location | Source |
| --- | --- | --- |
| ZINC (12k subset) | `data/zinc/` | Downloaded automatically on first run |
| Cycle counting | `data/count_cycle/` | Included; dataset from [I2-GNN (Huang et al., 2023)](https://github.com/GraphPKU/I2GNN) |
| Directed patterns | `data/pattern/simple/` | Included; synthetic directed multigraphs from [Egressy et al., 2024](https://arxiv.org/abs/2306.11586)|
| DrugOOD | `data/drugood/` | `python download_drugood.py`|

## Reproducing the paper

Run the ZINC, cycle counting and pattern experiments from the experiment's own directory. Their runners take these flags:

- `--config_dirpath` and `--config_name`: which config to load.
- `--gpu_id N`: which GPU to use.
- `--seed N`: the random seed.
- `--out_dirpath` (ZINC and cycle counting only): where checkpoints and logs go, overriding the config.
- `--testing`: turns off Weights & Biases logging.

Without `--testing`, runs log to the W&B project named by `wandb_project` in the config, so run `wandb login` first or edit that field.

### ZINC

```bash
cd zinc
python runner.py --config_dirpath ../configs/zinc --config_name DIAL.yaml --seed 0 --num_seeds 4
```


### Directed pattern detection

```bash
cd patterns
python runner.py --config_dirpath ../configs/pattern --config_name dial.yaml  --gpu_id 0
python runner.py --config_dirpath ../configs/pattern --config_name pearl.yaml --gpu_id 0
```

Each command trains seeds 0–4 with $K = 4$. To compare $K \in \{1, 2, 4, 8, 12, 16\}$, run `sweep_runner.py` with `dial_sweep.yaml` or `pearl_sweep.yaml`.

### Cycle counting — I2GNN (`configs/cycle_counts_i2gnn/`)

```bash
cd cycle_count_i2gnn
python runner.py --config_dirpath ../configs/cycle_counts_i2gnn --config_name pearl.yaml --gpu_id 0
python runner.py --config_dirpath ../configs/cycle_counts_i2gnn --config_name dial.yaml --gpu_id 0
```

Expects the I2GNN `count_cycle` dataset under `data/count_cycle`. Use `--cycles 3,4` to train only some cycle lengths, and `--num_seeds` to change the number of seeds (default 3).

### DrugOOD

Download the data first with `pip install gdown && python download_drugood.py`. Then run DIAL on each of the three LBAP-core-IC50 domain splits:

```bash
cd drugood
python runner.py --config_dirpath ../configs/drugood --config_name dial_assay.yaml    --dataset assay    --seed 0
python runner.py --config_dirpath ../configs/drugood --config_name dial_scaffold.yaml --dataset scaffold --seed 0
python runner.py --config_dirpath ../configs/drugood --config_name dial_size.yaml     --dataset size     --seed 0
```

Each command trains one seed. It reports ROC-AUC on the in-distribution and out-of-distribution validation and test sets.

## Acknowledgements

This codebase is adapted from [SPE (Huang et al., 2024)](https://github.com/Graph-COM/SPE) and [PEARL (Kanatsoulis et al., 2025)](https://github.com/ehejin/Pearl-PE). The cycle counting data and preprocessing come from [I2-GNN](https://github.com/GraphPKU/I2GNN).
