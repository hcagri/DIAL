# Rebuttal experiment jobs

The scripts assume that the repository is available at
`/itet-stor/speltonen/net_scratch/dial`, that the Conda installation is under
`/itet-stor/speltonen/net_scratch/conda`, and that the environment is named
`dial`. Override `PROJECT_ROOT`, `LOG_ROOT`, `CONDA_ROOT`, or
`CONDA_ENVIRONMENT` through `sbatch --export` if needed.

Create the log directory before submitting jobs because Slurm opens its output
files before the job script runs:

```bash
mkdir -p /itet-stor/speltonen/net_scratch/dial/log
```

The GPU request uses `#SBATCH --gres=gpu:1`. Confirm that this is the correct
syntax for the target partition before submitting.

## CIFAR10 superpixels

On the first run, submit one pilot task and wait for it to download and process
the shared dataset:

```bash
sbatch --array=0 slurm/superpixels.sbatch
```

Then submit the remaining jobs:

```bash
sbatch --array=1-8 slurm/superpixels.sbatch
```

Tasks 0--2 run the baseline, 3--5 run R-PEARL, and 6--8 run DIAL; each block
uses seeds 0--2.

## ZINC with a GPS-style backbone

Likewise, use one task to prepare the shared ZINC download before launching the
remaining array:

```bash
sbatch --array=0 slurm/zinc_gps.sbatch
sbatch --array=1-9 slurm/zinc_gps.sbatch
```

Tasks 0--4 run GPS and 5--9 run DIAL+GPS, using seeds 0--4.

## Cycle counting with a GPS-style backbone

```bash
sbatch slurm/cycle_gps.sbatch
```

Tasks 0--11 run GPS and 12--23 run DIAL+GPS. Within each block, the jobs run
3-, 4-, 5-, and 6-cycle counting, with seeds 0--2 for each target.

Slurm stdout and stderr are written directly under `log/`. Checkpoints and the
per-run training logs are written under `log/results/`.
