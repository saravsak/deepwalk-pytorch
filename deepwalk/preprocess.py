# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, data, window=5, unk='<UNK>', max_vocab = 20000):
        self.window = window
        self.unk = unk
        self.data = data
        self.max_vocab = max_vocab

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self):
        max_vocab = self.max_vocab
        print("building vocab...")
        step = 0
        self.wc = {self.unk: 1}
        for line in self.data:
            step += 1
            if not step % 1000:
                print("working on {}kth line".format(step // 1000), end='\r')
            sent = line
            for word in sent:
                self.wc[word] = self.wc.get(word, 0) + 1
        print("")
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        print("build done")
        return self.idx2word, self.word2idx, self.vocab, self.wc

    def convert(self):
        print("converting corpus...")
        step = 0
        data = []
        for line in self.data:
            step += 1
            if not step % 1000:
                print("working on {}kth line".format(step // 1000), end='\r')
            
            sent = []
            for word in line:
                if word in self.vocab:
                    sent.append(word)
                else:
                    sent.append(self.unk)
            for i in range(len(sent)):
                iword, owords = self.skipgram(sent, i)
                data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        print("")
        #pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print("conversion done")
        return data

if __name__ == '__main__':
    args = parse_args()
    import numpy as np
    data = np.array(np.random.randint(0,10312, (10312, 80)),dtype=str)
    preprocess = Preprocess(data,window=args.window, unk=args.unk)
    preprocess.build(max_vocab=args.max_vocab)
    preprocess.convert()
