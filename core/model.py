import torch.nn as nn
import torch
import os
import json
import numpy as np
import math
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))

    def extract_feature(self, x):
        return self.base(x)


class FE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        x = self.fc3(x)
        return x


def simplecnn(hidden_dim, n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, hidden_dim], output_dim=n_classes)


class TextCNN_FE(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TextCNN_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=100,
                kernel_size=(size, emb_size)
            )
            for size in [3, 4, 5]
        ])
        self.relu = nn.ReLU()

    def forward(self, text):
        embeddings = self.embedding(text).unsqueeze(1)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = torch.cat(pooled, dim=1)
        return flattened


class TextCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_size):
        super(TextCNN, self).__init__()
        self.base = TextCNN_FE(vocab_size, emb_size)
        self.classifier = Classifier(300, n_classes)

    def forward(self, x):
        return self.classifier((self.base(x)))


def textcnn(n_classes):
    with open(os.path.join("data", 'word_map.json'), 'r') as j:
        word_map = json.load(j)
        vocab_size = len(word_map)
    return TextCNN(n_classes, vocab_size, 256)


class GFK(nn.Module):
    def __init__(self, level, nfeat, nlayers, nhidden, nclass, dropoutC, dropoutM, bias, sole=True):
        super(GFK, self).__init__()
        self.nfeat = nfeat
        self.level = level + 1
        self.comb = Combination(nfeat, self.level, dropoutC, sole)
        self.mlp = MLP(nfeat, nlayers, nhidden, nclass, dropoutM, bias)

    def extract_feature(self, x):
        x = self.comb(x)
        x = self.mlp.extract_feature(x)

        return x

    def forward(self, x):
        x = self.comb(x)
        x = self.mlp(x)

        return x


class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


class MLP(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, bias):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers - 2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def extract_feature(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)

        return x

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fcs[-1](x)
        return x


class Combination(nn.Module):

    def __init__(self, channels, level, dropout, sole=False):
        super().__init__()
        self.dropout = dropout
        self.K = level
        self.comb_weight = nn.Parameter(torch.ones((1, level, 1)))
        self.nfeat = channels
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / self.K
        TEMP = np.random.uniform(bound, bound, self.K * self.nfeat)
        self.comb_weight = nn.Parameter(torch.FloatTensor(TEMP).view(-1, self.K, self.nfeat))

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x
