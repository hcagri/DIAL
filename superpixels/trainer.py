import logging
import math
import os
import random
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.utils import get_laplacian

from root import root
from src.data_utils.dataloader import DataLoader
from src.mlp import MLP
from src.model import PEARL_GNN_Model, construct_model


class Trainer:
    """Train a graph classifier on an official superpixel split."""

    def __init__(self, cfg: DictConfig, gpu_id: Optional[int]) -> None:
        set_seed(int(cfg.seed))
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.device = torch.device(
            f"cuda:{gpu_id}"
            if gpu_id is not None and torch.cuda.is_available()
            else "cpu"
        )

        configured_name = str(cfg.get("dataset_name", "CIFAR10"))
        dataset_name = {
            "cifar10": "CIFAR10",
            "mnist": "MNIST",
        }.get(configured_name.lower(), configured_name)
        cfg.dataset_name = dataset_name
        dataset_root = root(os.path.join("data", "superpixels"))
        train_dataset = GNNBenchmarkDataset(
            root=dataset_root, name=dataset_name, split="train"
        )
        val_dataset = GNNBenchmarkDataset(
            root=dataset_root, name=dataset_name, split="val"
        )
        test_dataset = GNNBenchmarkDataset(
            root=dataset_root, name=dataset_name, split="test"
        )

        # These values depend on the selected GNNBenchmarkDataset and therefore
        # intentionally override the placeholders in the YAML configuration.
        cfg.n_node_types = int(train_dataset.num_node_features)
        cfg.target_dim = int(train_dataset.num_classes)
        sample = train_dataset[0]
        edge_attr = getattr(sample, "edge_attr", None)
        cfg.n_edge_types = (
            1
            if edge_attr is None or edge_attr.ndim == 1
            else int(edge_attr.size(-1))
        )
        if bool(cfg.wandb) and wandb.run is not None:
            wandb.config.update(
                {
                    "dataset_name": dataset_name,
                    "n_node_types": cfg.n_node_types,
                    "n_edge_types": cfg.n_edge_types,
                    "target_dim": cfg.target_dim,
                },
                allow_val_change=True,
            )

        num_workers = int(cfg.get("num_workers", 4))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.train_batch_size),
            shuffle=True,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.val_batch_size),
            shuffle=False,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.val_batch_size),
            shuffle=False,
            num_workers=num_workers,
        )

        model_kwargs = {
            "device": str(self.device),
            "residual": bool(cfg.residual),
            "bn": bool(cfg.batch_norm),
            "sn": bool(cfg.graph_norm),
            "feature_type": "continuous",
        }
        self.model: PEARL_GNN_Model = construct_model(
            cfg,
            (self.create_mlp, self.create_mlp_ln),
            **model_kwargs,
        )
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.scheduler = self._cosine_schedule_with_warmup(
            int(cfg.n_warmup_steps), self.n_total_steps
        )
        self.criterion = nn.CrossEntropyLoss()

        self.logger = logging.getLogger(str(uuid.uuid4()))
        self.logger.setLevel(logging.INFO)
        out_dir = root(
            os.path.join(str(cfg.out_dirpath), dataset_name.lower())
        )
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_path = os.path.join(
            out_dir, f"best_seed{self.seed}_{timestamp}.pt"
        )
        handler = logging.FileHandler(
            os.path.join(out_dir, f"train_seed{self.seed}_{timestamp}.log")
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                "%Y/%m/%d %H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

        self.num_samples = int(cfg.num_samples)
        self.basis = bool(cfg.basis)

    def train(self) -> tuple[float, float]:
        self.logger.info("Configuration:\n%s", OmegaConf.to_yaml(self.cfg))
        self.logger.info(
            "Total parameters: %d",
            sum(parameter.numel() for parameter in self.model.parameters()),
        )

        best_val_accuracy = -math.inf
        best_epoch = 0
        best_state = None

        for epoch in range(1, int(self.cfg.n_epochs) + 1):
            train_loss = self.train_epoch()
            val_accuracy = self.evaluate(
                self.val_loader, evaluation_seed=10_000
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                torch.save(best_state, self.checkpoint_path)

            learning_rate = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
                f"val {val_accuracy:.4f} | best val {best_val_accuracy:.4f} | "
                f"lr {learning_rate:.6f}"
            )
            self.logger.info(
                "epoch=%d loss=%.6f val_accuracy=%.6f "
                "best_val_accuracy=%.6f lr=%.8f",
                epoch,
                train_loss,
                val_accuracy,
                best_val_accuracy,
                learning_rate,
            )
            if bool(self.cfg.wandb) and wandb.run is not None:
                wandb.log(
                    {
                        f"seed_{self.seed}/train_loss": train_loss,
                        f"seed_{self.seed}/val_accuracy": val_accuracy,
                        f"seed_{self.seed}/best_val_accuracy": best_val_accuracy,
                        f"seed_{self.seed}/learning_rate": learning_rate,
                    }
                )

        if best_state is None:
            raise RuntimeError("Training completed without a validation checkpoint")

        self.model.load_state_dict(best_state)
        best_test_accuracy = self.evaluate(
            self.test_loader, evaluation_seed=20_000
        )
        if bool(self.cfg.wandb) and wandb.run is not None:
            wandb.run.summary[f"seed_{self.seed}/best_epoch"] = best_epoch
            wandb.run.summary[
                f"seed_{self.seed}/best_val_accuracy"
            ] = best_val_accuracy
            wandb.run.summary[
                f"seed_{self.seed}/test_accuracy"
            ] = best_test_accuracy

        return best_val_accuracy, best_test_accuracy

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = self._prepare_batch(batch)
            probes = self._prepare_encoding(batch)
            targets = batch.y.view(-1).long()

            self.optimizer.zero_grad()
            logits = self.model(batch, probes)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * targets.numel()

        return total_loss / len(self.train_loader.dataset)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, evaluation_seed: int = 0) -> float:
        self.model.eval()
        num_correct = 0
        num_examples = 0
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + evaluation_seed)

        for batch in loader:
            batch = self._prepare_batch(batch)
            probes = self._prepare_encoding(batch, generator)
            targets = batch.y.view(-1).long()
            logits = self.model(batch, probes)
            num_correct += int(
                (logits.argmax(dim=-1) == targets).sum().item()
            )
            num_examples += targets.numel()

        return num_correct / num_examples

    def _prepare_batch(self, batch):
        batch = batch.to(self.device)
        if batch.x is None:
            raise ValueError("The selected superpixel dataset has no node features.")
        batch.x = batch.x.float()
        if getattr(batch, "edge_attr", None) is not None:
            batch.edge_attr = batch.edge_attr.float()
            if batch.edge_attr.ndim == 1:
                batch.edge_attr = batch.edge_attr.unsqueeze(-1)
        return batch

    def _prepare_encoding(self, batch, generator=None):
        """Attach one sparse batch Laplacian and generate full-batch probes."""
        pe_method = str(self.cfg.pe_method).lower()
        if pe_method == "none":
            return None
        if pe_method not in {"dial", "pearl"}:
            raise ValueError(f"Unsupported pe_method for superpixels: {pe_method}")

        lap_index, lap_values = get_laplacian(
            batch.edge_index,
            normalization="sym",
            num_nodes=batch.num_nodes,
        )
        batch.Lap = torch.sparse_coo_tensor(
            lap_index,
            lap_values.to(dtype=batch.x.dtype),
            size=(batch.num_nodes, batch.num_nodes),
            device=self.device,
        ).coalesce()

        if self.basis:
            probes = torch.eye(
                batch.num_nodes, device=self.device, dtype=batch.x.dtype
            )
        elif pe_method == "dial":
            probes = torch.empty(
                batch.num_nodes,
                self.num_samples,
                device=self.device,
                dtype=batch.x.dtype,
            )
            probes.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        else:
            probes = torch.randn(
                batch.num_nodes,
                self.num_samples,
                device=self.device,
                dtype=batch.x.dtype,
                generator=generator,
            )

        # The encoder accepts per-graph probe lists for legacy datasets. Here a
        # one-element list denotes one tensor aligned with the full sparse batch.
        return [probes]

    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * int(self.cfg.n_epochs)

    def create_mlp(self, in_dims: int, out_dims: int) -> MLP:
        return MLP(
            int(self.cfg.n_mlp_layers),
            in_dims,
            int(self.cfg.mlp_hidden_dims),
            out_dims,
            bool(self.cfg.mlp_use_bn),
            str(self.cfg.mlp_activation),
            float(self.cfg.mlp_dropout_prob),
        )

    def create_mlp_ln(
        self, in_dims: int, out_dims: int, use_bias: bool = True
    ) -> MLP:
        return MLP(
            int(self.cfg.n_mlp_layers),
            in_dims,
            int(self.cfg.mlp_hidden_dims),
            out_dims,
            bool(self.cfg.mlp_use_ln),
            str(self.cfg.mlp_activation),
            float(self.cfg.mlp_dropout_prob),
            norm_type="layer",
            NEW_BATCH_NORM=True,
            use_bias=use_bias,
        )

    def _cosine_schedule_with_warmup(
        self, num_warmup_steps: int, num_training_steps: int
    ) -> optim.lr_scheduler.LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return max(
                    1e-6,
                    float(current_step) / float(max(1, num_warmup_steps)),
                )
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
