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

For the GPS-backbone comparison, run the control and DIAL variants:

```bash
cd zinc
python runner.py --config_dirpath ../configs/zinc --config_name gps.yaml --seed 0 --num_seeds 1
python runner.py --config_dirpath ../configs/zinc --config_name dial_gps.yaml --seed 0 --num_seeds 1
```

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

The GPS comparison can be split by cycle length and seed for parallel cluster jobs:

```bash
cd cycle_count_i2gnn
python runner.py --config_dirpath ../configs/cycle_counts_i2gnn --config_name gps.yaml --cycles 3 --seed 0 --num_seeds 1
python runner.py --config_dirpath ../configs/cycle_counts_i2gnn --config_name dial_gps.yaml --cycles 3 --seed 0 --num_seeds 1
```

### Continuous node features (`configs/superpixels/`)

The CIFAR10 superpixel experiment uses continuous color and coordinate features and the official train/validation/test splits. Run the task-only, R-PEARL, and DIAL variants from the repository root:

```bash
python -m superpixels.runner --config_name baseline --num_seeds 1
python -m superpixels.runner --config_name rpearl --num_seeds 1
python -m superpixels.runner --config_name dial --num_seeds 1
```

The probe configurations initially use 32 samples for quick turnaround. Override settings on the command line, for example with `num_samples=64`, or use the smaller MNIST superpixel benchmark as a runtime fallback with `--dataset_name MNIST`.
Run one pilot job first so that the dataset is downloaded and processed before launching parallel jobs on a shared filesystem.

### Slurm

Array jobs for the rebuttal experiments are documented in
[`slurm/README.md`](slurm/README.md). They write Slurm output, checkpoints, and
per-run training logs under the repository's `log/` directory.

## Attribution

This code framework is adapted from the SPE repository by [Huang et al., 2024](https://github.com/Graph-COM/SPE) and [Kanatsoulis et al., 2025](https://github.com/ehejin/Pearl-PE).
