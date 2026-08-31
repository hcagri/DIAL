import logging
from datetime import datetime
import os
import random
import uuid
from typing import TextIO, Optional, List, Dict, Any

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from src.other_models import PNANet
from src.data_utils.dataloader import DataLoader 
from torch_geometric.utils import get_laplacian, to_dense_adj

from root import root
from src.mlp import MLP
from src.model import PEARL_GNN_Model, construct_model
from src.schema import Schema
import cycle_count_i2gnn.data_processing as dp

from collections import defaultdict

def print_parameter_count_by_module(model):
    module_params = defaultdict(int)
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.split('.')[0]  # Get the top-level module name
            num_params = param.numel()
            module_params[module_name] += num_params
            total_params += num_params
    
    print("\nParameter count by module:\n")
    for module, num_params in module_params.items():
        print(f"{module}: {num_params} parameters")

    print(f"\nTotal parameters: {total_params}")
    return total_params

def print_parameter_count(model):
    total_params = 0
    print("Detailed parameter count:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"{name}: {num_params} parameters")
            total_params += num_params
    print(f"\nTotal parameters: {total_params}")

    return total_params


class Trainer:
    cfg: Schema
    model: PEARL_GNN_Model
    train_loader: List
    val_loader: List
    test_loader: List
    optimizer: optim.Adam
    criterion: nn.BCEWithLogitsLoss
    logger: logging.Logger
    val_writer: TextIO
    curr_epoch: int
    curr_batch: int

    def __init__(self, cfg: Schema, target: int, gpu_id: Optional[int]) -> None:
        set_seed(cfg.seed)
        self.seed = cfg.seed
        self.cfg = cfg
        self.target = target
        cfg.out_dirpath = root(cfg.out_dirpath)

        root_path = root("data")

        train_dataset = dp.dataset_random_graph(dataname='count_cycle', root = root_path,processed_name='processed', transform=self.MyTransform,split='train')
        val_dataset = dp.dataset_random_graph(dataname='count_cycle', root = root_path,processed_name='processed', transform=self.MyTransform, split='val')
        test_dataset = dp.dataset_random_graph(dataname='count_cycle', root = root_path,processed_name='processed', transform=self.MyTransform, split='test')


        # normalize target
        y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
        mean = y_train_val.mean(dim=0)
        std = y_train_val.std(dim=0)
        train_dataset.data.y = (train_dataset.data.y - mean) / std
        val_dataset.data.y = (val_dataset.data.y - mean) / std
        test_dataset.data.y = (test_dataset.data.y - mean) / std
        print('For target %d, Mean = %.3f, Std = %.3f' % (self.target, mean[self.target], std[self.target]))

        self.std = std

        self.test_loader = DataLoader(test_dataset, batch_size=cfg.val_batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)


        # construct model after loading dataset
        kwargs = {} # only works for pe-dim=8
        kwargs["deg"] = None
        self.device = torch.device("cpu" if gpu_id is None else f"cuda:{gpu_id}")
        kwargs["device"] = str(self.device)
        kwargs["residual"] = cfg.residual
        kwargs["bn"] = cfg.batch_norm
        kwargs["sn"] = cfg.graph_norm
        kwargs["feature_type"] = "continuous"


        if cfg.base_model == 'PNAnet':
                d = degree(train_dataset.edge_index[1], num_nodes=train_dataset.num_nodes).long()
                deg_hist = torch.bincount(d, minlength=int(d.max()) + 1)
                
                self.model = PNANet(
                in_dim=1, 
                hidden_dim=cfg.base_hidden_dims, 
                out_dim=1, 
                deg=deg_hist, 
                num_layers=cfg.n_base_layers, 
                dropout=0.1,
                residual=cfg.residual,
            )
        else:
            self.model = construct_model(cfg, (self.create_mlp, self.create_mlp_ln), **kwargs)
    

        self.model.to(self.device)

        # Construct auxiliary training objects
        param_groups = self.get_param_groups()

        self.optimizer = optim.Adam(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.9, patience=10, min_lr=0.00001)
        # self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)

        self.criterion = nn.L1Loss(reduction="mean")
        self.metric = nn.L1Loss(reduction='sum')

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
        self.wandb = cfg.wandb

        # Miscellaneous
        self.curr_epoch = 1
        self.curr_batch = 1
        self.basis = cfg.basis
        self.num_samples = cfg.num_samples


    def train(self, seed="", name='RAND_200_TEST') -> None:
        print(self.seed, "SEED")
        self.logger.info("Configuration:\n" + OmegaConf.to_yaml(self.cfg))
        print(f"Total parameters: {sum(param.numel() for param in self.model.parameters())}")

        print_parameter_count_by_module(self.model)
        #print_parameter_count(self.model)

        self.logger.info(f"Total parameters: {sum(param.numel() for param in self.model.parameters())}")
        self.logger.info(f"Total training steps: {self.n_total_steps}")
        self.logger.info(
            "Optimizer groups:\n" + "\n".join(group["name"] for group in self.optimizer.param_groups) + "\n")

        best_val_loss = float('inf')
        best_test_loss = float('inf')
        best_test_loss_normed = float('inf')
        for self.curr_epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss  = self.evaluate(self.val_loader)
            self.scheduler.step(val_loss)
            test_loss  = self.evaluate(self.test_loader)
            test_loss_normed = test_loss / self.std[self.target]

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.wandb:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss, 'lr': lr})
            if val_loss.sum() < best_val_loss:
                best_val_loss = val_loss
                best_test_loss, best_test_loss_normed = test_loss, test_loss_normed
                if self.wandb:
                    wandb.run.summary["best_val_loss" + str(seed)] = best_val_loss
                    wandb.run.summary["Test MAE:" + str(seed)] = best_test_loss
                    wandb.run.summary["Test MAE norm:" + str(seed)] = best_test_loss_normed
                    wandb.run.summary["best_training_loss" + str(seed)] = train_loss
                
                print(f"EPOCH {self.curr_epoch}: Val MAE {best_val_loss}, Test MAE: {best_test_loss}, Test MAE norm: {best_test_loss_normed}")

        return best_test_loss, best_test_loss_normed


    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for self.curr_batch, batch in enumerate(self.train_loader, 1):
            total_loss += self.train_batch(batch)

        return total_loss / self.train_loader.dataset.y.size(0)

    
    def train_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
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
        y_pred = self.model(batch, W_list)               # [B]
        loss = self.criterion(y_pred, batch.y.view(-1, 1))   # [1]
        loss.backward()
        self.optimizer.step()

        loss = loss.item()

        # self.scheduler.step()

        return loss * batch.y.size(0)


    def evaluate_batch(self, batch: Batch) -> float:
        batch.to(device(self.model))
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
                print("EVAL BATCH, i, LAP|W: ", i, batch.Lap[i].shape, W.shape)
            W_list.append(W)

        with torch.no_grad():
            y_pred = self.model(batch, W_list)

        return self.metric(y_pred, batch.y.view(-1, 1))

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        total_count = 0

        for batch in eval_loader:
            loss = self.evaluate_batch(batch)
            self.logger.info(f"Evaluating... Epoch: {self.curr_epoch}, batch: {self.curr_batch}")
        
            total_loss += loss
            total_count += batch.y.size(0)
        
        avg_loss = total_loss / max(total_count, 1) * self.std[self.target]

        self.val_writer.write(f"Epoch: {self.curr_epoch}\t Loss: {avg_loss}\n")
        self.val_writer.flush()

        return avg_loss


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
            self.cfg.pearl_act, self.cfg.mlp_dropout_prob, norm_type="layer", NEW_BATCH_NORM=True, use_bias=use_bias
        )

    def get_lap(self, instance: Data) -> Data:
        n = instance.num_nodes[0]
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
    
    def MyTransform(self, g):
        if getattr(g, 'x', None) is None:
            N = g.y.shape[0] if getattr(g, 'y', None) is not None else int(g.edge_index.max()) + 1
            # Assign constant features (all ones) to each node
            g.x = torch.ones((N, 1), dtype=torch.float)
        g.y = g.y[:, int(self.target)]
        g = self.get_lap(g)
        return g


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
