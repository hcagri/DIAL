import logging
import os
from datetime import datetime
import random
import uuid
from typing import TextIO, Optional, List, Dict, Any
import math 
import warnings

# Show dial-GIN warning only once (not per forward pass)
warnings.filterwarnings('once', message='Beaware you are still using dial-GIN')

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchmetrics.classification import BinaryAUROC

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

from src.data_utils.dataloader import DataLoader 
from torch_geometric.utils import get_laplacian, to_dense_adj

# models
from root import root
from src.mlp import MLP
from src.model import PEARL_GNN_Model, GINEBaseModel, construct_model
from src.schema import Schema


class Trainer:
    cfg: Schema
    model: PEARL_GNN_Model
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: optim.Adam
    criterion: nn.BCEWithLogitsLoss
    metric: BinaryAUROC
    logger: logging.Logger
    val_writer: TextIO
    curr_epoch: int
    curr_batch: int

    def __init__(self, cfg: Schema, gpu_id: Optional[int]) -> None:
        set_seed(cfg.seed)

        # Initialize configuration
        self.cfg = cfg
        cfg.out_dirpath = root(cfg.out_dirpath)

        transform = self.get_lap
        dataset = PygGraphPropPredDataset(
                        name='ogbg-molhiv', 
                        root=root("data"), 
                        transform=transform,
                        pre_transform=None
                    )

        split_idx = dataset.get_idx_split()

        self.evaluator = Evaluator('ogbg-molhiv')

        self.train_loader = DataLoader(dataset[split_idx["train"]], 
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(dataset[split_idx["valid"]], 
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=4
        )
        self.test_loader = DataLoader(dataset[split_idx["test"]], 
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=4
        )
        # construct model after loading dataset
        kwargs = {}
        kwargs["device"] = f"cuda:{gpu_id}"
        kwargs["residual"] = cfg.residual
        kwargs["bn"] = cfg.batch_norm
        kwargs["sn"] = cfg.graph_norm
        kwargs["feature_type"] = "molhiv"
        
        # Compute degree histogram for PNA (required by PNAConv scalers)
        deg1 = self._compute_deg(dataset[split_idx["train"]])
        
        self.model = construct_model(cfg, (self.create_mlp, self.create_mlp_ln), deg1=deg1, **kwargs)
        print(self.model)
        self.model.to("cpu" if gpu_id is None else f"cuda:{gpu_id}")

        param_groups = self.get_param_groups()
        self.optimizer = optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = self.get_cosine_schedule_with_warmup(cfg.n_warmup_steps, self.n_total_steps)
        self.device = torch.device(f'cuda:{gpu_id}')
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([6.0], device=self.device))
        
        self.metric = dataset.eval_metric

        # Set up logger and writer
        name = str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        os.makedirs(cfg.out_dirpath, exist_ok=True)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handler = logging.FileHandler(os.path.join(cfg.out_dirpath, f"train_logs_{self.run_timestamp}.txt"))
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y/%m/%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.val_writer = open(os.path.join(cfg.out_dirpath, f"evaluate_logs_{self.run_timestamp}.txt"), "a")

        # Set up WandB
        cfg.__dict__['num_params'] = sum(param.numel() for param in self.model.parameters())
        print("num params: ", cfg.__dict__['num_params'])

        # Miscellaneous
        self.curr_epoch = 1
        self.curr_batch = 1
        self.basis = cfg.basis
        self.num_samples = cfg.num_samples
        self.seed = cfg.seed

    def train(self) -> None:
        self.logger.info("Configuration:\n" + OmegaConf.to_yaml(self.cfg))
        self.logger.info(f"Total parameters: {sum(param.numel() for param in self.model.parameters())}")
        self.logger.info(f"Total training steps: {self.n_total_steps}")
        self.logger.info(
            "Optimizer groups:\n" + "\n".join(group["name"] for group in self.optimizer.param_groups) + "\n")

        best_valid_perf = -1E6
        for self.curr_epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self.train_epoch()
            valid_perf = self.evaluate(self.val_loader)
            test_perf = self.evaluate(self.test_loader)

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f"Epoch {self.curr_epoch:3d} | Loss: {train_loss:.4f} | Valid: {valid_perf:.4f} | Test: {test_perf:.4f} | LR: {lr:.6f}")

            if valid_perf > best_valid_perf:
                best_valid_perf = valid_perf
                best_test_perf = test_perf
                wandb.run.summary["best_valid_perf" + str(self.seed)] = best_valid_perf
                wandb.run.summary["best_test_perf" + str(self.seed)] = best_test_perf
                wandb.run.summary["best_training_loss" + str(self.seed)] = train_loss

                print(f"  ★ New Best! Val: {best_valid_perf:.4f} | Test: {best_test_perf:.4f}")

        return best_test_perf

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for step, batch in enumerate(self.train_loader):

            batch = batch.to(self.device)
            skip_epoch = batch.x.shape[0] == 1 or batch.batch[-1] == 0

            if skip_epoch:
                pass

            y = batch.y
            y = y.to(torch.float32)
            
            ### Forward pass on batch
            self.optimizer.zero_grad()
            W_list = []
            for i in range(len(batch.Lap)):
                if self.basis:
                    W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                else:
                    if self.cfg.pe_method == 'dial':
                        W = np.random.choice([-1, 1], 
                                            size=(batch.Lap[i].shape[0],self.num_samples))
                        W = torch.from_numpy(W).float().to(self.device)
                    else:
                        W = torch.randn(batch.Lap[i].shape[0],self.num_samples).to(self.device) #BxNxM
                if len(W.shape) < 2:
                    print("TRAIN BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
                W_list.append(W)
            y_pred = self.model(batch, W_list)

            loss = self.criterion(y_pred.view(-1, 1), y)
            loss = loss.mean()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * y.shape[0]

        return total_loss / len(self.train_loader.dataset)


    def evaluate(self, loader):
        self.model.eval()

        y_true_list = []
        y_pred_list = []

        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            skip_epoch = batch.x.shape[0] == 1

            if skip_epoch:
                continue

            with torch.no_grad():
                W_list = []
                for i in range(len(batch.Lap)):
                    if self.basis:
                        W = torch.eye(batch.Lap[i].shape[0]).to(self.device)
                    else:
                        if self.cfg.pe_method == 'dial':
                            W = np.random.choice([-1, 1], 
                                                size=(batch.Lap[i].shape[0], self.num_samples))
                            W = torch.from_numpy(W).float().to(self.device)
                        else:
                            W = torch.randn(batch.Lap[i].shape[0], self.num_samples).to(self.device)
                    W_list.append(W)
                
                pred = self.model(batch, W_list).view(-1, 1)
                y = batch.y.to(torch.float32)

            y_true_list.append(y.detach().cpu())
            y_pred_list.append(pred.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_pred = torch.cat(y_pred_list, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res = self.evaluator.eval(input_dict)
        return res['rocauc']

    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.cfg.n_epochs

    def create_mlp(self, in_dims: int, out_dims: int) -> MLP:
        return MLP(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_bn,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob
         )

    def create_mlp_ln(self, in_dims: int, out_dims: int, use_bias=True) -> MLP:
        return MLP(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_ln,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob, norm_type="layer"
        )

    def _compute_deg(self, dataset) -> torch.Tensor:
        """Compute degree histogram from dataset for PNAConv scalers."""
        max_degree = 0
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        
        # Compute histogram
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg

    def get_lap(self, instance: Data) -> Data:
        n = instance.num_nodes
        L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
        L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)
        instance.Lap = L
        return instance

    def get_param_groups(self) -> List[Dict[str, Any]]:
        return [{
            "name": name,
            "params": [param],
            "weight_decay": 0.0 if "bias" in name else self.cfg.weight_decay
        } for name, param in self.model.named_parameters()]

    def lr_lambda(self, curr_step: int) -> float:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
        """
        if curr_step < self.cfg.n_warmup_steps:
            return curr_step / max(1, self.cfg.n_warmup_steps)
        else:
            return max(0.0, (self.n_total_steps - curr_step) / max(1, self.n_total_steps - self.cfg.n_warmup_steps))
    
    def get_cosine_schedule_with_warmup(self, num_warmup_steps: int, num_training_steps: int,
            num_cycles: float = 0.5, last_epoch: int = -1):
        """
        Implementation by Huggingface:
        https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

        Create a schedule with a learning rate that decreases following the values
        of the cosine function between the initial lr set in the optimizer to 0,
        after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.
        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`float`, *optional*, defaults to 0.5):
                The number of waves in the cosine schedule (the defaults is to just
                decrease from the max value to 0 following a half-cosine).
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.
        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch)


def set_seed(seed: int) -> None:
    """
    Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_utils.py#L83
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device