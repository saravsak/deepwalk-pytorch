# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np
import sys

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, data, ws=None):
        #data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(data, idx2word, wc, e_dim=128, name='word2vec', n_negs = 5, conti=False, cuda=False, epoch = 1, ss_t=1e-5,mb=4096, weights=False, save_dir='./output'):
    #idx2word = pickle.load(open(os.path.join(data_dir, 'idx2word.dat'), 'rb'))
    #wc = pickle.load(open(os.path.join(data_dir, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = wf if weights else None
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=e_dim)
    modelpath = os.path.join(save_dir, '{}.pt'.format(name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=n_negs, weights=weights)
    if os.path.isfile(modelpath) and conti:
        sgns.load_state_dict(t.load(modelpath))
    if cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    optimpath = os.path.join(save_dir, '{}.optim.pt'.format(name))
    if os.path.isfile(optimpath) and conti:
        optim.load_state_dict(t.load(optimpath))
    for epoch in range(1, epoch + 1):
        flag = False
        dataset = PermutedSubsampledCorpus(data)
        dataloader = DataLoader(dataset, batch_size=mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        losses = []
        prev_loss = 0
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            losses.append(loss.item())
            prev_loss = loss.item()
            if mean(losses[-10:]) < sys.epsilon:
                flag = True
                break
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
        if flag:
            break
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    #pickle.dump(idx2vec, open(os.path.join(data_dir, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(save_dir, '{}.pt'.format(name)))
    t.save(optim.state_dict(), os.path.join(save_dir, '{}.optim.pt'.format(name)))
    return idx2vec


if __name__ == '__main__':
    train()
