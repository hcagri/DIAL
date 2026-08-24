#!/bin/sh
#SBATCH --job-name=saku-size-31-dec
#SBATCH --account=ewi-st-dis
#SBATCH --qos=medium 
#SBATCH --partition=general        # Request partition.
#SBATCH --exclude=influ[2-5],insy[15-16]
#SBATCH --time=16:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=16G
#SBATCH --mail-type=END            # Set mail type to 'END' to receive a mail when the job finishes. %j is the Slurm jobId
#SBATCH --output=./slurm-%x-%j.out
#SBATCH --error=./slurm-%x-%j.err

# Increase file descriptor limit
ulimit -n 65536

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/tudelft.net/staff-umbrella/ScalableGraphLearning/apptainer"
export APPTAINER_NAME="saku_v2.sif"

# Generate random seed between 1 and 100
RANDOM_SEED=$(( (RANDOM % 100) + 1 ))
echo "Using random seed: $RANDOM_SEED"

nvidia-smi

srun apptainer exec \
  --nv \
  -B /home/nfs/cbilgi/projects/saku-pe/PEARL/drugood:/home/$USER/projects/saku-pe/PEARL/drugood \
  $APPTAINER_ROOT/$APPTAINER_NAME \
  python runner.py --config_dirpath ../configs/drugood/size --config_name RSAKU_GINE_GIN.yaml --seed $RANDOM_SEED --dataset size

