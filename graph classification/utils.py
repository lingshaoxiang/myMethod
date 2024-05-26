import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim as optim
from sklearn import preprocessing as sk_prep

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument("--max_epoch", type=int, default=2000, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")

    parser.add_argument("--max_epoch_f", type=int, default=2000, help="number of evaluation epochs")
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")

    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_p", type=float, default=0.5)

    parser.add_argument("--in_drop", type=float, default=0.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--activation", type=str, default="prelu")

    parser.add_argument("--alpha", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--tau", type=float, default=2.0)

    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")

    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--use_cfg", action="store_true", default=False)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def create_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return None


def load_best_configs(args, path):
    with open(path, "r", encoding='utf-8') as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args