## Installation

Create a conda environment with Python 3.9 and install PyTorch (CUDA build matching your system), then:

```bash
pip install -r requirements.txt
pip install -r req_pyg.txt
```

`req_pyg.txt` targets PyTorch Geometric wheels for `torch-2.2.0+cu118`; adjust the `--find-links` URL if you use a different Torch/CUDA version.

## How to run experiments

Configs live under `configs/`. Run each experiment from its corresponding directory. Add `--testing` to disable online WandB, and `--gpu_id N` to select a GPU.

### ZINC (`configs/zinc/`)

```bash
cd zinc
python runner.py --config_dirpath ../configs/zinc --config_name PEARL_GINE_GIN.yaml --seed 0
python runner.py --config_dirpath ../configs/zinc --config_name DIAL.yaml --seed 0 --num_seeds 1
```

ZINC data is downloaded automatically to `data/zinc` on first run (`use_subset: True` in these configs).

### Pattern detection (`configs/pattern/`)

These configs are intended for a sweep over `pearl_k` via `sweep_runner.py`:

```bash
cd patterns
python sweep_runner.py --config_dirpath ../configs/pattern --config_name pearl_sweep.yaml --gpu_id 0
python sweep_runner.py --config_dirpath ../configs/pattern --config_name dial_sweep.yaml --gpu_id 0
```

Expects `data/pattern/simple/{train,val,test}.pt`.

### Cycle counting — I2GNN (`configs/cycle_counts_i2gnn/`)

```bash
cd cycle_count_i2gnn
python runner.py --config_dirpath ../configs/cycle_counts_i2gnn --config_name pearl.yaml --gpu_id 0
python runner.py --config_dirpath ../configs/cycle_counts_i2gnn --config_name dial.yaml --gpu_id 0
```

Expects the I2GNN `count_cycle` dataset under `data/count_cycle`.

## Attribution

This code framework is adapted from the SPE repository by [Huang et al., 2024](https://github.com/Graph-COM/SPE) and [Kanatsoulis et al., 2025](https://github.com/ehejin/Pearl-PE).