import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from torch.nn.parameter import Parameter
from torch.nn import init

from tqdm import tqdm
torch.cuda.set_device(1)

global_aa = list("ACDEFGHIKLMNPQRSTVWY")

def pad_sequence(seq, val, max_len):
    "Pad a single sequence with a val."
    diff = max_len - len(seq)
    remain = diff%2
    sq = val*int((diff-remain)/2)
    new_string=str(sq)+seq+str(sq)
    new_string+=val*int(remain)
    return new_string

def readcsv_class(path):
    data = []
    lengthdistr = [0 for i in range(21)]
    filtered = 0
    grv = np.zeros(3)
    
    seq = []
    tgt = []
    with open(path+"/data.tsv", 'rt') as fin:
        for i, line in enumerate(fin):
            #if '*' in line: continue
            line = line.rstrip('\n').split('\t')[0]
            if len(line)>20: print(True)
            line = pad_sequence(line, "J", 20)
            seq.append(line)
            #lengthdistr[len(line)] += 1
    with open(path+"/data.target", 'rt') as fin:
        for i, line in enumerate(fin):
            line = list(map(float, line.rstrip('\n').split("\t")))
            tgt.append(np.array(line))
    data = [z for z in zip(seq, tgt)]
    print (len(data), lengthdistr)

    return np.random.permutation(data)

def readcsv_reg(path):
    data = []
    lengthdistr = [0 for i in range(21)]
    filtered = 0
    grv = np.zeros(3)
    
    seq = []
    tgt = []
    with open(path+"/data.tsv", 'rt') as fin:
        for i, line in enumerate(fin):
            line = line.rstrip('\n').split('\t')[1]
            line = pad_sequence(line, "J", 20)
            seq.append(line)
            lengthdistr[len(line)] += 1
    with open(path+"/data.target", 'rt') as fin:
        for i, line in enumerate(fin):
            line = float(line.rstrip('\n'))
            tgt.append(line)
    data = [z for z in zip(seq, tgt)]
    print (len(data), lengthdistr)

    return np.random.permutation(data)

def readcsv_reg_test(path):
    data = []
    lengthdistr = [0 for i in range(21)]
    filtered = 0
    grv = np.zeros(3)
    
    seq = []
    tgt = []
    with open(path+"/test.tsv", 'rt') as fin:
        for i, line in enumerate(fin):
            line = line.rstrip('\n').split('\t')[1]
            line = pad_sequence(line, "J", 20)
            seq.append(line)
            lengthdistr[len(line)] += 1
    with open(path+"/test.target", 'rt') as fin:
        for i, line in enumerate(fin):
            line = float(line.rstrip('\n'))
            tgt.append(line)
    data = [z for z in zip(seq, tgt)]
    print (len(data), lengthdistr)

    return np.random.permutation(data)

def process_reg(data):
    onehot = {}
    onehot['J'] = np.zeros(20)
    for aa, vec in zip(global_aa, np.eye(20)):
        onehot[aa] = vec
        
    inputs = []
    labels = []
    ranges = []
    lg10 = np.log(10)
    for aa, frac in tqdm(data, position=0, leave=True):
        extra = 40-len(aa)
        r = math.floor(extra/2)
        l = math.ceil(extra/2)
        aa = np.array([onehot[c] for c in ('J'*r) + aa + ('J'*l)]).T
        minmax = (40-l-20, r) #This range is inclusive: (try l=r=10)
        
        inputs.append(aa)
        labels.append(float(frac))
        ranges.append(minmax)
    return inputs, labels, ranges

def process_class(data):
    onehot = {}
    onehot['J'] = np.zeros(20)
    for aa, vec in zip(global_aa, np.eye(20)):
        onehot[aa] = vec
        
    inputs = []
    labels = []
    ranges = []
    lg10 = np.log(10)
    for aa, frac in tqdm(data, position=0, leave=True):
        extra = 40-len(aa)
        r = math.floor(extra/2)
        l = math.ceil(extra/2)
        aa = np.array([onehot[c] for c in ('J'*r) + aa + ('J'*l)]).T
        minmax = (40-l-20, r) #This range is inclusive: (try l=r=10)
        
        inputs.append(aa)
        labels.append(frac)
        ranges.append(minmax)
    return inputs, labels, ranges

def toTensorDataset(data):
    torchds = torch.utils.data.TensorDataset(torch.tensor(np.array(data[0]), dtype = torch.float32),
                                             torch.tensor(np.array(data[1]), dtype = torch.float32),
                                             torch.tensor(np.array(data[2]), dtype = torch.int64))
    return torchds