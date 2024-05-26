import logging
from tqdm import tqdm
import numpy as np

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from datetime import datetime
import os

from utils import (
    build_args,
    set_random_seed,
    load_best_configs,
)
from data_utils import load_graph_classification_dataset
from edcoder import PreModel


def graph_classification_evaluation(model, pooler, dataloader, device):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, logger=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g, _ = batch
            batch_g = batch_g.to(device)

            feat = batch_g.ndata["attr"]
            model.train()
            loss, loss_dict = model(batch_g, feat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

            
def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels

def main(args):
    device = args.device if args.device >= 0 else "cpu"

    graphs, (num_features, num_classes) = load_graph_classification_dataset(args.dataset, deg4feat=args.deg4feat)
    args.num_features = num_features

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True)
    eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)


    if args.pooling == "mean":
        pooler = AvgPooling()
    elif args.pooling == "max":
        pooler = MaxPooling()
    elif args.pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError
    

    currentTime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_dir = 'results/' + currentTime + '_' + args.dataset 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    output_file = open(f"{save_dir}/{args.dataset}_final_result.txt", "w")
    output_file.write(f'Settings: {args} \n')

    acc_list = []
    for run in range(args.runs):
        print(f"####### Run {run} #######")
        set_random_seed(args.seed)

        model = PreModel(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            activation=args.activation,
            dropout=args.in_drop,
            residual=args.residual,
            norm=args.norm,
            mask_rate=args.mask_rate,
            drop_p=args.drop_p,
            alpha=args.alpha,
            tau=args.tau,
            concat_hidden=args.concat_hidden
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        model = pretrain(model, (train_loader, eval_loader), optimizer, args.max_epoch, device, scheduler)

        save_path_model_pretrain = f'{save_dir}/pretrain_model_{args.dataset}_run_{run}.pt'
        torch.save(model.state_dict(), save_path_model_pretrain)

        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(model, pooler, eval_loader, device)
        acc_list.append(test_f1)

        output_file.write(f'run: {run}, final_acc: {test_f1:.4f}\n')


    final_acc_mean, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc_mean:.4f}±{final_acc_std:.4f}")

    output_file.write(f'all runs end, final_acc: {final_acc_mean:.4f}±{final_acc_std:.4f}')
    output_file.close()


if __name__ == "__main__":

    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    main(args)