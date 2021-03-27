
import os
import json
import argparse
import pprint
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from bnaf import *
from tqdm import trange
from tqdm import tqdm
from data.generate2d import sample2d, energy2d
from data.lognormal import data_lognormal
from torch.utils.data import TensorDataset, DataLoader

def create_model(args, verbose=False):
    
    flows = []
    for f in range(args.flows):
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(1 * args.hidden_dim,
                                       1 * args.hidden_dim, dim=1))
            layers.append(Tanh())

        flows.append(
            BNAF(*([MaskedWeight(1, 1 * args.hidden_dim, dim=1), Tanh()] + \
                   layers + \
                   [MaskedWeight(1 * args.hidden_dim, 1, dim=1)]),\
                 res='gated' if f < args.flows - 1 else False
            )
        )

        if f < args.flows - 1:
            flows.append(Permutation(1, 'flip'))

    model = Sequential(*flows).to(args.device)
    
    if verbose:
        print('{}'.format(model))
        print('Parameters={}, n_dims={}'.format(sum((p != 0).sum() 
                                                    if len(p.shape) > 1 else torch.tensor(p.shape).item() 
                                                    for p in model.parameters()), 1))
    
    return model


def compute_log_p_x(model, x_mb):
    y_mb, log_diag_j_mb = model(x_mb)
    # y_mb = torch.sigmoid(y_mb)
    # log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb)*100000000).log_prob(y_mb).sum(-1)
    log_p_y_mb = torch.distributions.Uniform(torch.zeros_like(y_mb), torch.ones_like(y_mb)*1000000).log_prob(y_mb).sum(-1)

    return log_p_y_mb + log_diag_j_mb

import math
def train_density1d(model, dataloader, optimizer, scheduler, args):
    # iterator = trange(args.steps, smoothing=0, dynamic_ncols=True)

    best_loss = 1e20

    for epoch in range(args.steps):
        tloss = 0
        cnt = 0
        t = tqdm(dataloader, smoothing=0, ncols=80)
        for x in t:
            cnt += 1
            x_mb = x[0].to(args.device)
            loss = - compute_log_p_x(model, x_mb).mean()
            tloss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            scheduler.step(loss)
        print('epoch {}, loss {}'.format(epoch, tloss/cnt))
            # iterator.set_postfix(loss='{:.2f}'.format(loss.data.cpu().numpy()), refresh=False)

        if math.isnan(tloss):
            print('loss nan')
            exit()
        if tloss < best_loss:
            print('Saving..')
            best_loss = tloss
            save(model, optimizer, os.path.join(args.load or args.path, 'checkpoint.pt'))

def test_density1d(model, dataloader, args):
    # iterator = trange(args.steps, smoothing=0, dynamic_ncols=True)
    model.eval()
    cnt = 0
    t = tqdm(dataloader, smoothing=0, ncols=80)
    results = []
    with torch.no_grad():
        for x in t:
            cnt += 1
            x_mb = x[0].to(args.device)
            y_mb, log_diag_j_mb = model(x_mb)
            results.append(y_mb)
        return torch.cat(results)
        # iterator.set_postfix(loss='{:.2f}'.format(loss.data.cpu().numpy()), refresh=False)


def compute_kl(model, args):
    d_mb = torch.distributions.Normal(torch.zeros((args.batch_dim, 2)).to(args.device),
                                      torch.ones((args.batch_dim, 2)).to(args.device))
    y_mb = d_mb.sample()
    x_mb, log_diag_j_mb = model(y_mb)
    log_p_y_mb = d_mb.log_prob(y_mb).sum(-1)
    return log_p_y_mb - log_diag_j_mb + energy2d(args.dataset, x_mb)         + (torch.relu(x_mb.abs() - 6) ** 2).sum(-1)


def train_energy1d(model, optimizer, scheduler, args):
    iterator = trange(args.steps, smoothing=0, dynamic_ncols=True)
    for epoch in iterator:

        loss = compute_kl(model, args).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(loss)

        iterator.set_postfix(loss='{:.2f}'.format(loss.data.cpu().numpy()), refresh=False)


def load(model, optimizer, path):
    print('Loading dataset..')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
def save(model, optimizer, path):
    print('Saving model..')
    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)


def plot_density2d(model, args, limit=4, step=0.01):

    grid = torch.Tensor([[a, b] for a in np.arange(-limit, limit, step) for b in np.arange(-limit, limit, step)])
    grid_dataset = torch.utils.data.TensorDataset(grid.to(args.device))
    grid_data_loader = torch.utils.data.DataLoader(grid_dataset, batch_size=10000, shuffle=False)

    prob = torch.cat([torch.exp(compute_log_p_x(model, x_mb)).detach() 
                      for x_mb, in grid_data_loader], 0)
    
    prob = prob.view(int(2 * limit / step), int(2 * limit / step)).t()

    if args.reduce_extreme:
        prob = prob.clamp(max=prob.mean() + 3 * prob.std())

    plt.figure(figsize=(12, 12))
    plt.imshow(prob.cpu().data.numpy(), extent=(-limit, limit, -limit, limit))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if args.savefig:
        plt.savefig(os.path.join(args.load or args.path, '{}.jpg'.format(datetime.datetime.now())))
    else:
        plt.show()

        
def plot_energy2d(model, args, limit=4, step=0.05, resolution=(10000, 10000)):

    a = np.hstack([model(torch.randn(resolution[0], 2).to(args.device))[0].t().cpu().data.numpy()
                   for _ in trange(resolution[1])])
    
    H, _, _ = np.histogram2d(a[0], a[1], bins=(np.arange(-limit, limit, step), np.arange(-limit, limit, step)))

    plt.figure(figsize=(12, 12))
    plt.imshow(H.T, interpolation='gaussian')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if args.savefig:
        plt.savefig(os.path.join(args.load or args.path, '{}.jpg'.format(datetime.datetime.now())))
    else:
        plt.show()

## test ##

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='lognormal',
                        choices=['lognormal'])
    parser.add_argument('--experiment', type=str, default='density1d',
                        choices=['density1d', 'energy1d'])

    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--batch_dim', type=int, default=200)
    parser.add_argument('--clip_norm', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=15)
    
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--decay', type=float, default=0.5)

    parser.add_argument('--flows', type=int, default=1)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--savefig', action='store_true')
    parser.add_argument('--reduce_extreme', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.save = True
    d_tensors = data_lognormal('/home/nandcui/data').all

    x = d_tensors.clone()
    indices = torch.randperm(x.shape[0])[:100000]

    x = x[indices]

    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)


    print('Arguments:')
    pprint.pprint(args.__dict__)

    args.path = os.path.join('checkpoint', '{}{}_layers{}_h{}_flows{}_{}'.format(
        args.expname + ('_' if args.expname != '' else ''),
        args.dataset, args.layers, args.hidden_dim, args.flows,
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    if (args.save or args.savefig) and not args.load:
        print('Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
    
    print('Creating BNAF model..')
    model = create_model(args, verbose=True)

    print('Creating optimizer..')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    
    print('Creating scheduler..')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay,
                                                           patience=args.patience,
                                                           min_lr=5e-4, verbose=True,
                                                           threshold_mode='abs')
    
    if args.load:
        load(model, optimizer, os.path.join(args.load, 'checkpoint.pt'))
        
    print('Training..')
    if args.experiment == 'density1d':
        train_density1d(model, dataloader, optimizer, scheduler, args)
    else:
        train_energy1d(model, optimizer, scheduler, args)


    dataset = TensorDataset(d_tensors)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

    results = test_density1d(model,dataloader,args)

    np.savetxt('lognormal_100_latent_uniform.txt', results.detach().cpu().numpy(),  fmt='%.18f')

    # if args.save:
    #     print('Saving..')
    #     save(model, optimizer, os.path.join(args.load or args.path, 'checkpoint.pt'))

    # print('Plotting..')
    # if args.experiment == 'density2d':
    #     plot_density2d(model, args)
    # elif args.experiment == 'energy2d':
    #     plot_energy2d(model, args)


if __name__ == '__main__':
    main()
