import torch.nn
from utils import *
from data_utils import load_dataset
from edcoder import PreModel
from datetime import datetime
import os

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, save_path_model, patience):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    best = 1e9
    best_epoch = 0
    cnt_wait = 0
    for epoch in range(1, 1 + max_epoch):
        model.train()

        optimizer.zero_grad()

        loss, loss_dict = model(graph, x)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if loss < best:
            best = loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path_model)
            cnt_wait = 0
        else:
            cnt_wait += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Best_epoch: {best_epoch:02d}, '
                f'Loss: {loss:.4f}, ')
                
            print('***************')
        if cnt_wait == patience:
            print('Early stop at {}'.format(epoch))
            break

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(save_path_model))

    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"

    graph, num_features, num_classes = load_dataset(args.dataset)
    args.num_features = num_features

    currentTime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_dir = 'results/' + currentTime + '_' + args.dataset 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    output_file = open(f"{save_dir}/{args.dataset}_final_result.txt", "w")
    output_file.write(f'Settings: {args} \n')

    acc_list = []
    estp_acc_list = []
    for run in range(args.runs):
        set_random_seed(args.seed)
        print(f"####### Run {run} #######")

        model = PreModel(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=1,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
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

        x = graph.ndata["feat"]

        save_path_model_linear = f'{save_dir}/linear_best_model_{args.dataset}_run_{run}.pth'
        save_path_model_pretrain = f'{save_dir}/pretrain_model_{args.dataset}_run_{run}.pt'
        model = pretrain(model, graph, x, optimizer, args.max_epoch, device, scheduler, save_path_model_pretrain, args.patience)

        model = model.to(device)
        model.eval()
        final_acc, estp_acc, best_val_epoch = node_classification_evaluation(model, graph, x, num_classes, args.lr_f,
                                                             args.weight_decay_f, args.max_epoch_f, device, save_path_model_linear)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        
        output_file.write(f'run: {run}, final_acc: {final_acc:.4f}, early-stopping_acc at best_val_epoch {best_val_epoch}: {estp_acc:.4f}\n')


    final_acc_mean, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc_mean, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc_mean:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc_mean:.4f}±{estp_acc_std:.4f}")

    output_file.write(f'all runs end, final_acc: {final_acc_mean:.4f}±{final_acc_std:.4f}, early-stopping_acc: {estp_acc_mean:.4f}±{estp_acc_std:.4f}')
    output_file.close()



if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    main(args)
