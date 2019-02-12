import os
import pickle
import random
import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from .model import Word2Vec, SGNS

import pdb
from .preprocess import Preprocess
import pdb

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


class ModWord2Vec():
    def __init__(self,sentences=None,
            corpus_file=None, 
            size=100, 
            alpha=0.025, 
            window=5, 
            min_count=5, 
            max_vocab_size=None, 
            sample=0.001, 
            seed=1, 
            workers=3, 
            min_alpha=0.0001, 
            sg=0, 
            hs=0, 
            negative=5, 
            ns_exponent=0.75, 
            cbow_mean=1, 
            hashfxn=None, 
            iter=5, 
            null_word='<UNK>', 
            trim_rule=None, 
            sorted_vocab=1, 
            batch_words=10000, 
            compute_loss=False, 
            callbacks=(), 
            max_final_vocab=None):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()

    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        for self.epoch in range(1, self.epoch + 1):
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            for iword, owords in pbar:
                loss = sgns(iword, owords)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_postfix(loss=loss.item())
        idx2vec = model.ivectors.weight.data.cpu().numpy()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)



if __name__ == "__main__":
        data = np.array(np.random.randint(0,13210, size=(13210, 80)),str)
        w2v = ModWord2Vec(data)
        w2v.save_emb("embedding.npy",13210)
