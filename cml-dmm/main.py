import os
import random
import torch
import argparse

import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn

from torch import nn, optim
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

from datasets import load_dataset
from PyTorchCML import losses, models, samplers, regularizers, evaluators, trainers


def main(args):
    # set seed for reproduct
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    # load dataset using args.dataset
    # download movielens dataset
    dataset = args.dataset
    data_path = args.data_path

    n_user, n_item, train_set, test_set = load_dataset(dataset, data_path)
    # to torch.Tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set = torch.LongTensor(train_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    lr = 1e-3
    n_dim = args.n_dim
    margin = args.margin
    if args.model == 'cml':
        model = models.CollaborativeMetricLearning(n_user, n_item, n_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_dict = {
                'sumtriplet': losses.SumTripletLoss(margin=margin),
                'sumtriplet_dml': losses.SumTripletLossDML(margin=margin),
                'mintriplet': losses.MinTripletLoss(margin=margin),
                'mintriplet_dml': losses.MinTripletLossDML(margin=margin),
                'bpr': losses.BprLoss()
            }
    elif args.model == 'mf':
        model = models.MatrixFactorization(n_user, n_item, n_dim).to(device)
    sampler = samplers.BaseSampler(train_set, n_user, n_item, device=device, strict_negative=True, n_neg_samples=args.num_neg)

    score_function_dict = {
        "nDCG" : evaluators.ndcg,
        "MAP" : evaluators.average_precision,
        "Recall": evaluators.recall,
        "HitRatio": evaluators.hit 
    }
    evaluator = evaluators.UserwiseEvaluator(test_set, score_function_dict, ks=[1, 5, 10])
    trainer = trainers.BaseTrainer(model, optimizer, sampler, criterion_dict, args=args)

    
    trainer.fit(n_batch=256, n_epoch=args.epoch, valid_evaluator = evaluator, valid_per_epoch=10)
    print(trainer.valid_scores)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CML experimental settings.')
    parser.add_argument('--model', type=str, default='mf')
    parser.add_argument('--loss', type=str, default='mintriplet')
    parser.add_argument('--loss_type', type=str, default='base')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='movielens')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--num_neg', type=int, default=10)
    args = parser.parse_args()
    main(args)