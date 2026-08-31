"""
Hyperparameter Sweep Script for MolHIV Experiments
This script runs a grid search over specified hyperparameters.
"""

import sys
sys.path.append('../')  # ensure correct path dependency

import argparse
import csv
import itertools
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import wandb

from hydra import initialize, compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.schema import Schema
from trainer import Trainer
from root import root


# Define the hyperparameter grid
HYPERPARAMETER_GRID = {
    'pearl_k': [1, 2, 4, 8],
    'num_samples': [32, 64, 120],
    'lr': [1e-3, 5e-4], 
    'base_hidden_dims': [64, 128],  
}


def get_grid_combinations() -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters from the grid."""
    keys = list(HYPERPARAMETER_GRID.keys())
    values = list(HYPERPARAMETER_GRID.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def run_single_experiment(
    base_cfg: Schema,
    hyperparams: Dict[str, Any],
    gpu_id: int,
    seeds: List[int],
    testing: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment with given hyperparameters across multiple seeds.
    
    Args:
        base_cfg: Base configuration object
        hyperparams: Dictionary of hyperparameters to override
        gpu_id: GPU device ID
        seeds: List of random seeds to run
        testing: Whether this is a test run
    
    Returns:
        Dictionary with results including mean and std of test performance
    """
    # Override configuration with sweep hyperparameters
    for key, value in hyperparams.items():
        setattr(base_cfg, key, value)
    
    # Update wandb run name to include hyperparameters
    hyperparam_str = "_".join([f"{k}={v}" for k, v in hyperparams.items()])
    base_cfg.wandb_run_name = f"SWEEP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hyperparam_str}"
    
    # Initialize WandB for this sweep run
    if base_cfg.wandb:
        if not testing:
            wandb.login(key="")  # use your own WandB key
        wandb.init(
            mode="disabled" if testing else "online",
            dir=root("."),
            project="ICML-2026-sweep-molhiv",
            name=base_cfg.wandb_run_name,
            config={**base_cfg.__dict__, **hyperparams},
            tags=["sweep", f"pearl_k={hyperparams['pearl_k']}"],
            reinit=True  # Allow multiple wandb.init() calls
        )
    
    # Run across multiple seeds
    test_rocaucs = []
    for seed in seeds:
        base_cfg.seed = seed
        print(f"\n{'='*60}")
        print(f"Running seed {seed} with hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        trainer = Trainer(base_cfg, gpu_id)
        best_test_perf = trainer.train()
        test_rocaucs.append(best_test_perf)
        print(f"Seed {seed} - Test ROCAUC: {best_test_perf:.4f}")
    
    # Compute statistics
    mean_rocauc = np.mean(test_rocaucs)
    std_rocauc = np.std(test_rocaucs)
    
    # Log summary to WandB
    if base_cfg.wandb:
        wandb.run.summary["Test(ROCAUC) Mean"] = mean_rocauc
        wandb.run.summary["Test(ROCAUC) Std"] = std_rocauc
        for key, value in hyperparams.items():
            wandb.run.summary[f"hp_{key}"] = value
        wandb.finish()  # Properly close this run
    
    print(f"\n{'='*60}")
    print(f"RESULTS for {hyperparam_str}:")
    print(f"  Mean ROCAUC: {mean_rocauc:.4f} ± {std_rocauc:.4f}")
    print(f"  Individual seeds: {[f'{x:.4f}' for x in test_rocaucs]}")
    print(f"{'='*60}\n")
    
    # Return results
    results = {
        **hyperparams,
        'mean_rocauc': mean_rocauc,
        'std_rocauc': std_rocauc,
        'seed_results': test_rocaucs
    }
    
    return results


def initialize_csv_log(csv_path: str, hyperparam_keys: List[str]) -> None:
    """Initialize CSV file with headers."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = hyperparam_keys + ['mean_rocauc', 'std_rocauc', 'seed_0', 'seed_1', 'seed_2', 'seed_3', 'seed_4']
        writer.writerow(headers)


def append_to_csv_log(csv_path: str, results: Dict[str, Any], hyperparam_keys: List[str]) -> None:
    """Append results to CSV file."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [results[key] for key in hyperparam_keys]
        row.extend([results['mean_rocauc'], results['std_rocauc']])
        row.extend(results['seed_results'])
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for MolHIV experiments")
    parser.add_argument("--config_dirpath", type=str, required=True,
                        help="Path to config directory (relative to molhiv/)")
    parser.add_argument("--config_name", type=str, required=True,
                        help="Name of the base config file (without .yaml)")
    parser.add_argument("--gpu_id", type=int, required=False, default=0,
                        help="GPU device ID")
    parser.add_argument("--testing", action="store_true",
                        help="Run in testing mode (disables WandB)")
    parser.add_argument("--output_dir", type=str, default="sweep_results",
                        help="Directory to save sweep results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = root(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this sweep
    sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"sweep_results_{sweep_timestamp}.csv")
    
    # Get all hyperparameter combinations
    combinations = get_grid_combinations()
    total_runs = len(combinations)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP CONFIGURATION")
    print("="*80)
    print(f"Base config: {args.config_name}")
    print(f"Total hyperparameter combinations: {total_runs}")
    print(f"Seeds per combination: 5 (0, 1, 2, 3, 4)")
    print(f"Total training runs: {total_runs * 5}")
    print("\nHyperparameter grid:")
    for key, values in HYPERPARAMETER_GRID.items():
        print(f"  {key}: {values}")
    print(f"\nResults will be saved to: {csv_path}")
    print("="*80 + "\n")
    
    # Initialize CSV log
    hyperparam_keys = list(HYPERPARAMETER_GRID.keys())
    initialize_csv_log(csv_path, hyperparam_keys)
    
    # Seeds to use for each experiment
    seeds = [0, 1, 2, 3, 4]
    
    # Run sweep
    all_results = []
    for idx, hyperparams in enumerate(combinations, 1):
        print(f"\n{'#'*80}")
        print(f"SWEEP ITERATION {idx}/{total_runs}")
        print(f"{'#'*80}")
        
        # Clear any existing Hydra instance to avoid conflicts
        GlobalHydra.instance().clear()
        
        # Initialize Hydra for this iteration
        cs = ConfigStore.instance()
        cs.store(name="schema", node=Schema)
        initialize(version_base=None, config_path=args.config_dirpath)
        
        # Compose base configuration
        dict_cfg: DictConfig = compose(config_name=args.config_name)
        cfg: Schema = OmegaConf.to_object(dict_cfg)
        
        # Run experiment with current hyperparameters
        results = run_single_experiment(
            base_cfg=cfg,
            hyperparams=hyperparams,
            gpu_id=args.gpu_id,
            seeds=seeds,
            testing=args.testing
        )
        
        # Log results
        all_results.append(results)
        append_to_csv_log(csv_path, results, hyperparam_keys)
        
        # Print progress
        print(f"\nCompleted {idx}/{total_runs} combinations")
        print(f"Current best: {max(all_results, key=lambda x: x['mean_rocauc'])}")
    
    # Final summary
    print("\n" + "="*80)
    print("SWEEP COMPLETED!")
    print("="*80)
    
    # Find best configuration
    best_result = max(all_results, key=lambda x: x['mean_rocauc'])
    print("\nBest hyperparameters found:")
    for key in hyperparam_keys:
        print(f"  {key}: {best_result[key]}")
    print(f"\nBest Mean ROCAUC: {best_result['mean_rocauc']:.4f} ± {best_result['std_rocauc']:.4f}")
    print(f"\nAll results saved to: {csv_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

