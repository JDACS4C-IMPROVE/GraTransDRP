import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gat_gcn_transformer import GAT_GCN_Transformer
from models.gat_gcn_transformer_ge_only import GAT_GCN_Transformer_ge_only
from models.gat_gcn_transformer_meth_only import GAT_GCN_Transformer_meth_only
from models.gat_gcn_transformer_meth_ge import GAT_GCN_Transformer_meth_ge
from models.gat_gcn_transformer_meth import GAT_GCN_Transformer_meth
from models.gat_gcn_transformer_ge import GAT_GCN_Transformer_ge
from models.gat_gcn_transformer_meth_ge_mut import GAT_GCN_Transformer_meth_ge_mut
from utils import *
import datetime
import argparse


from time import time
class Timer:
    """
    Measure runtime.
    """
    def __init__(self):
        self.start = time()

    def timer_end(self):
        self.end = time()
        time_diff = self.end - self.start
        return time_diff

    def display_timer(self, print_fn=print):
        time_diff = self.timer_end()
        if (time_diff)//3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
        else:
            print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if 'VAE' in model_st:
            #For variation autoencoder
            output, _, decode, log_var, mu = model(data)
            loss = weight_fn * loss_fn(
                output,
                data.y.view(-1, 1).float().to(device)) + loss_ae(
                    decode,
                    data.target_mut[:, None, :].float().to(device)) + torch.mean(
                        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
                        dim=0)
        elif 'AE' in model_st:
            output, _, decode = model(data)
            loss = weight_fn * loss_fn(output,
                                       data.y.view(-1, 1).float().to(device)) + loss_ae(
                                           decode,
                                           data.target_mut[:,
                                                           None, :].float().to(device))
        #For non-variational autoencoder
        else:
            output, _ = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data.x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader, model_st):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #Non-variational autoencoder
            if 'VAE' in model_st:
                #For variation autoencoder
                output, _, decode, log_var, mu = model(data)
            elif 'AE' in model_st:
                output, _, decode = model(data)
            #For non-variational autoencoder
            else:
                output, _ = model(data)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval,
         cuda_name):

    # ap -----
    # import pdb; pdb.set_trace()
    timer = Timer()
    if args.dim_red == "kpca":
        dim_red_str = "_KernelPCA"
        # val_scheme = "pca"
    elif args.dim_red == "pca":
        dim_red_str = "_PCA"
        # val_scheme = "cell_blind"
    elif args.dim_red == "isomap":
        dim_red_str = "_Isomap"
        # val_scheme = "drug_blind"

    # ap: The authors perform only mixed-set validation analysis
    set_str = "_mix"

    from pathlib import Path
    fdir = Path(__file__).resolve().parent
    if args.gout is not None:
        outdir = fdir/args.gout
    else:
        outdir = fdir/"results"
    os.makedirs(outdir, exist_ok=True)

    root = args.root
    # ap -----

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    for model in modeling:

        model_st = model.__name__
        dataset = 'GDSC'
        train_losses = []
        val_losses = []
        val_pearsons = []
        print('\nrunning on ', model_st + '_' + dataset)

        # processed_data_file_train = 'data/processed/' + dataset + '_train_mix' + '.pt'
        # processed_data_file_val = 'data/processed/' + dataset + '_val_mix' + '.pt'
        # processed_data_file_test = 'data/processed/' + dataset + '_test_mix' + '.pt'
        processed_data_file_train = root + '/processed/' + dataset + '_train' + set_str + dim_red_str + '.pt'
        processed_data_file_val = root + '/processed/' + dataset + '_val' + set_str + dim_red_str + '.pt'
        processed_data_file_test = root + '/processed/'+ dataset + '_test' + set_str + dim_red_str + '.pt'

        if ((not os.path.isfile(processed_data_file_train))
                or (not os.path.isfile(processed_data_file_val))
                or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            # train_data = TestbedDataset(root='data', dataset=dataset + '_train_mix')
            # val_data = TestbedDataset(root='data', dataset=dataset + '_val_mix')
            # test_data = TestbedDataset(root='data', dataset=dataset + '_test_mix')

            # import pdb; pdb.set_trace()
            train_data = TestbedDataset(root='data', dataset=dataset + '_train' + set_str + dim_red_str)
            val_data = TestbedDataset(root='data', dataset=dataset + '_val' + set_str + dim_red_str)
            test_data = TestbedDataset(root='data', dataset=dataset + '_test' + set_str + dim_red_str)

            # make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
            print("CPU/GPU: ", torch.cuda.is_available())

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            print(device)
            model = model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            best_mse = 1000
            best_pearson = 1
            best_epoch = -1

            # model_file_name = 'model_' + model_st + '_' + dataset + '.model'
            # result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
            # loss_fig_name = 'model_' + model_st + '_' + dataset + '_loss'
            # pearson_fig_name = 'model_' + model_st + '_' + dataset + '_pearson'

            model_file_name = outdir/('model_' + model_st + '_' + dataset + '_' + args.dim_red + '.model')
            result_file_name = outdir/('result_' + model_st + '_' + dataset + '_' + args.dim_red + '.csv')
            loss_fig_name = str(outdir/('model_' + model_st + '_' + dataset + '_' + args.dim_red + '_loss'))
            pearson_fig_name = str(outdir/('model_' + model_st + '_' + dataset + '_' + args.dim_red + '_pearson'))

            # import pdb; pdb.set_trace()
            for epoch in range(num_epoch):
                train_loss = train(model, device, train_loader, optimizer, epoch + 1,
                                   log_interval, model_st)

                G, P = predicting(model, device, val_loader, model_st)
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

                G_test, P_test = predicting(model, device, test_loader, model_st)
                ret_test = [
                    rmse(G_test, P_test),
                    mse(G_test, P_test),
                    pearson(G_test, P_test),
                    spearman(G_test, P_test)
                ]

                train_losses.append(train_loss)
                val_losses.append(ret[1])
                val_pearsons.append(ret[2])

                # Reduce Learning rate on Plateau for the validation loss
                scheduler.step(ret[1])

                if ret[1] < best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str, ret_test)))
                    best_epoch = epoch + 1
                    best_mse = ret[1]
                    best_pearson = ret[2]
                    print(' rmse improved at epoch ', best_epoch, '; best_mse:',
                          best_mse, model_st, dataset)
                else:
                    print(' no improvement since epoch ', best_epoch,
                          '; best_mse, best pearson:', best_mse, best_pearson, model_st,
                          dataset)

            draw_loss(train_losses, val_losses, loss_fig_name)
            draw_pearson(val_pearsons, pearson_fig_name)

            # ap -----
            # ap: Add to dump raw predictions
            G_test, P_test = predicting(model, device, test_loader, model_st)
            preds = pd.DataFrame({"True": G_test, "Pred": P_test})
            preds_file_name = f"preds_{args.dim_red}_{model_st}_{dataset}.csv"
            preds.to_csv(outdir/preds_file_name, index=False)

            # ap: Add code to calc and dump scores
            ccp_scr = pearson(G_test, P_test)
            rmse_scr = rmse(G_test, P_test)
            scores = {"ccp": ccp_scr, "rmse": rmse_scr}
            import json
            with open(outdir/f"scores_{args.dim_red}_{model_st}.json", "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)

            timer.display_timer()
            print(scores)
            print("Done.")
            # ap -----


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,
                        help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, '\
                             '2: Transformer_meth_mut, 3: Transformer_meth_ge, '\
                             '4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    parser.add_argument(
        '--train_batch',
        type=int,
        required=False,
        default=32,
        help='Batch size training set')
    parser.add_argument(
        '--val_batch',
        type=int,
        required=False,
        default=32,
        help='Batch size validation set')
    parser.add_argument(
        '--test_batch',
        type=int,
        required=False,
        default=32,
        help='Batch size test set')
    parser.add_argument(
        '--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument(
        '--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument(
        '--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    # ap -----
    parser.add_argument("--dim_red", type=str, default="kpca", choices=["kpca", "pca", "isomap"],
                        help="Dimensionality reduction for gene expression data.")
    parser.add_argument('--root', required=False, default="data", type=str,
                        help='Path to processed .pt files (default: data).')
    parser.add_argument('--gout', default=None, type=str,
                        help="Global outdir to dump all the resusts.")
    # ap -----

    args = parser.parse_args()

    # From parser.add_argument() above
    # help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, '
    #      '2: Transformer_meth_mut, 3: Transformer_meth_ge, '
    #      '4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth'
    modeling = [
        GAT_GCN_Transformer_meth_ge_mut,  # Transformer_ge_mut_meth
        GAT_GCN_Transformer_ge,           # Transformer_ge_mut
        GAT_GCN_Transformer_meth,         # Transformer_meth_mut
        GAT_GCN_Transformer_meth_ge,      # Transformer_meth_ge
        GAT_GCN_Transformer_ge_only,      # Transformer_ge
        GAT_GCN_Transformer,              # Transformer_mut
        GAT_GCN_Transformer_meth_only     # Transformer_meth
    ][args.model]
    model = [modeling]

    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name

    main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval,
         cuda_name)
