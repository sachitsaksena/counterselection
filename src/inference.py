import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from torch.nn.parameter import Parameter
from torch.nn import init

from tqdm import tqdm
torch.cuda.set_device(1)

def compute_pred_labels(ensemble_path, loader):
    models = [Seq32x1_16(), Seq32x2_16(), Seq64x1_16(), Seq_emb_32x1_16(), Seq32x1_16_filt3()]
    with torch.no_grad():
        overall_preds=[]
        pbar = tqdm(loader, position=0, leave=True)
        #pbar = loader
        for model in models:
            preds=[]
            truths=[]
            for (data, target, bounds) in pbar:
                #data = truncate(data, bounds).cuda()
                net = model
                net.load_state_dict(torch.load(ensemble_path+str(model).split("(")[0]+"/best.pt"))
                net = net.cuda()
                net = net.eval()
                data = data.cuda()
                pred = 1-np.array(list(map(np.argmax,net(data).cpu().numpy())))
                #preds.append( net(data).cpu().numpy() )
                preds.append(pred)
                truth=1-np.array(list(map(np.argmax, target.numpy())))
                truths.append(truth)
            preds = np.concatenate(preds)
            truths = np.concatenate(truths)
            overall_preds.append(preds)
        ensemble_preds = stats.mode(np.array(overall_preds), axis=0)[0]
    return ensemble_preds.reshape(-1), truths

def compute_pred_logits(ensemble_path, loader):
    models = [Seq32x1_16(), Seq32x2_16(), Seq64x1_16(), Seq_emb_32x1_16(), Seq32x1_16_filt3()]
    with torch.no_grad():
        overall_preds=[]
        pbar = tqdm(loader, position=0, leave=True)
        #pbar = loader
        for model in models:
            preds=[]
            truths=[]
            for (data, target, bounds) in pbar:
                #data = truncate(data, bounds).cuda()
                net = model
                net.load_state_dict(torch.load(ensemble_path+str(model).split("(")[0]+"/best.pt"))
                net = net.cuda()
                net = net.eval()
                data = data.cuda()
                pred = net(data).cpu().tolist()
                pred = [elt[0] for elt in pred]
                preds.append(pred)
                target = [elt[0] for elt in target]
                truths.append(target)
            preds = np.concatenate(preds)
            truths = np.concatenate(truths)
            overall_preds.append(preds)
        ensemble_preds = np.average(np.array(overall_preds), axis=0)
    return ensemble_preds, truths