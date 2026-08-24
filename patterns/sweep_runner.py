import sys
sys.path.append('../')  # ensure correct path dependency

import argparse
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import numpy as np

from src.schema import Schema
from trainer import Trainer
import torch
import wandb
from root import root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dirpath", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=False, default=0)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    # Initialize Hydra configuration
    cs = ConfigStore.instance()
    cs.store(name="schema", node=Schema)
    initialize(version_base=None, config_path=args.config_dirpath)
    dict_cfg: DictConfig = compose(config_name=args.config_name)
    
    # Convert to Schema object
    cfg: Schema = OmegaConf.to_object(dict_cfg)
    cfg.subset_size = args.subset_size

    # Initialize wandb for the sweep
    if cfg.wandb:
        wandb.login(key="")  # use your own Wandb key
        wandb.init(
            mode="disabled" if args.testing else "online",
            dir=root("."),
            project="ICML-2026",
            name=f"Sweep-{cfg.wandb_run_name}",
            config=cfg.__dict__,
            settings=wandb.Settings(code_dir=".")
        )

    # Define sweep parameters
    pearl_k_values = [1, 2, 4, 8, 12, 16]
    seeds = [0, 1, 2, 3, 4]
    tasks = ["deg-in", "deg-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]
    
    # Storage for results: {pearl_k: [macro_mean_seed0, macro_mean_seed1, ...]}
    sweep_results = {}
    
    print("=" * 80)
    print("Starting Hyperparameter Sweep over pearl_k")
    print("=" * 80)
    
    # Outer loop: iterate over pearl_k values
    for pearl_k in pearl_k_values:
        print(f"\n{'='*80}")
        print(f"Running experiments with pearl_k = {pearl_k}")
        print(f"{'='*80}\n")
        
        # Update configuration for current pearl_k
        cfg.pearl_k = pearl_k
        
        # Storage for this pearl_k's results across seeds
        test_f1_scores = []
        macro_scores = []
        
        # Inner loop: iterate over seeds
        for seed in seeds:
            print(f"\n--- pearl_k={pearl_k}, seed={seed} ---")
            cfg.seed = seed
            
            # Create trainer and run training
            trainer = Trainer(cfg, args.gpu_id)
            best_test_f1 = trainer.train()
            test_f1_scores.append(best_test_f1.cpu())
            
            # Calculate macro F1 for this seed
            macro_f1 = best_test_f1.mean().item() * 100
            macro_scores.append(macro_f1)
            print(f"Macro F1 for pearl_k={pearl_k}, seed={seed}: {macro_f1:.2f}%")
        
        # Calculate statistics across seeds for this pearl_k
        all_f1 = torch.stack(test_f1_scores, dim=0)
        mean_f1 = all_f1.mean(dim=0)
        std_f1 = all_f1.std(dim=0, unbiased=False)
        
        macro_mean = mean_f1.mean().item() * 100
        macro_std = np.std(macro_scores, ddof=0)
        
        # Store results
        sweep_results[pearl_k] = {
            'macro_mean': macro_mean,
            'macro_std': macro_std,
            'per_task_mean': mean_f1.tolist(),
            'per_task_std': std_f1.tolist(),
            'raw_macro_scores': macro_scores
        }
        
        print(f"\n[Summary] pearl_k={pearl_k}: Macro F1 = {macro_mean:.2f}±{macro_std:.2f}%")
        row = " | ".join(f"{n}: {100*m:.2f}±{100*s:.2f}%" 
                        for n, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()))
        print(f"Per-task: {row}\n")
        
        # Log to wandb (per pearl_k)
        if cfg.wandb:
            wandb.log({
                f"pearl_k_{pearl_k}/macro_mean": macro_mean,
                f"pearl_k_{pearl_k}/macro_std": macro_std,
            })
            
            # Create table for per-task results
            table = wandb.Table(columns=["Task", "Mean F1 (%)", "Std F1 (%)"])
            for task, m, s in zip(tasks, mean_f1.tolist(), std_f1.tolist()):
                table.add_data(task, m * 100, s * 100)
            wandb.log({f"pearl_k_{pearl_k}/per_task_f1": table})
    
    # ========================================
    # Generate LaTeX Table
    # ========================================
    print("\n" + "=" * 80)
    print("Sweep Complete - Generating LaTeX Table")
    print("=" * 80 + "\n")
    
    latex_table = generate_latex_table(sweep_results, pearl_k_values)
    print(latex_table)
    
    # Log LaTeX table to wandb
    if cfg.wandb:
        # Save as artifact
        artifact = wandb.Artifact("sweep_results", type="results")
        with artifact.new_file("latex_table.txt", mode="w") as f:
            f.write(latex_table)
        wandb.log_artifact(artifact)
        
        # Also log as text
        wandb.log({"latex_table": wandb.Html(f"<pre>{latex_table}</pre>")})
        
        # Create summary table
        summary_table = wandb.Table(columns=["pearl_k", "Macro Mean (%)", "Macro Std (%)"])
        for k in pearl_k_values:
            summary_table.add_data(k, 
                                  sweep_results[k]['macro_mean'], 
                                  sweep_results[k]['macro_std'])
        wandb.log({"sweep_summary": summary_table})
    
    if cfg.wandb:
        wandb.finish()
    
    print("\nSweep completed successfully!")


def generate_latex_table(sweep_results, pearl_k_values):
    """Generate a LaTeX table summarizing the sweep results."""
    
    tasks = ["deg-in", "deg-out", "C2", "C3", "C4", "C5", "C6", "S-G", "B-C"]
    
    latex = []
    latex.append("\\begin{table*}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Hyperparameter Sweep Results for pearl\\_k}")
    latex.append("\\label{tab:pearl_k_sweep}")
    
    # Build the column specification: pearl_k + 9 tasks + macro
    num_cols = len(tasks) + 2  # pearl_k + tasks + macro
    col_spec = "c|" + "c" * (num_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\hline")
    
    # Header row
    header = "\\textbf{pearl\\_k}"
    for task in tasks:
        # Escape special characters in task names
        task_escaped = task.replace("-", "\\text{-}")
        header += f" & \\textbf{{{task_escaped}}}"
    header += " & \\textbf{Macro F1}"
    latex.append(header + " \\\\")
    latex.append("\\hline")
    
    # Data rows
    for k in pearl_k_values:
        result = sweep_results[k]
        per_task_mean = result['per_task_mean']
        per_task_std = result['per_task_std']
        macro_mean = result['macro_mean']
        macro_std = result['macro_std']
        
        row = f"{k} L=6"
        for mean, std in zip(per_task_mean, per_task_std):
            row += f" & ${100*mean:.2f} \\pm {100*std:.2f}$"
        row += f" & ${macro_mean:.2f} \\pm {macro_std:.2f}$"
        latex.append(row + " \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    return "\n".join(latex)


if __name__ == "__main__":
    main()

