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

global_aa = list("ACDEFGHIKLMNPQRSTVWY")

def validate_logits(net, loader):
    preds = []
    truths = []
    with torch.no_grad():
        pbar = tqdm(loader, position=0, leave=True)
        #pbar = loader
        for (data, target, bounds) in pbar:
            #data = truncate(data, bounds).cuda()
            data = data.cuda()
            preds.append( net(data).cpu().numpy() )
            truths.append(target.numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        return preds, truths


def validate_labels(net, loader):
    preds = []
    truths = []
    with torch.no_grad():
        pbar = tqdm(loader, position=0, leave=True)
        #pbar = loader
        for (data, target, bounds) in pbar:
            #data = truncate(data, bounds).cuda()
            data = data.cuda()
            pred = 1-np.array(list(map(np.argmax, net(data).cpu().numpy())))
            #preds.append( net(data).cpu().numpy() )
            preds.append(pred)
            truth=1-np.array(list(map(np.argmax, target.numpy())))
            truths.append(truth)
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        return preds, truths

def train_ensemble(models, epochs, outpath, trainloader, trainloaderfast, testloader):
    clipper = WeightClipper(3)
    zeroout = WeightClipper(0)
    lossfunc = nn.BCELoss()
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    # train all 6 individual models 
    for model in models:
        best_val=0
        if not os.path.isdir(outpath+"/"+str(model).split("(")[0]):
            os.makedirs(outpath+"/"+str(model).split("(")[0])
        for i in range(epochs):
            losses = []
            net = model.cuda()
            optimizer = torch.optim.Adam(net.parameters())
            net = net.train()
            pbar = tqdm(trainloader, position=0, leave=True)
            for (data, target, bounds) in pbar:
                optimizer.zero_grad()
                #data = truncate(data, bounds).cuda()
                data = data.cuda()
                output = net(data)
                print(output)
                l1_norm = sum(p.abs().sum() for p in net.parameters())
                #print (l1_norm)
                #loss2 = (torch.mean(net.reg()**2))
                loss = lossfunc(output, target.cuda() ) + (0.0000001 * l1_norm)
                # + (loss2)

                loss.backward()
                optimizer.step()
                net.apply(clipper)
                #for submodel in net.nets():
                #    submodel.apply(clipper)
                closs = loss.item()
                losses.append(closs)
                #pbar.set_description("Batch {}: loss = {}, weight = {}".format(i, np.mean(losses), 0.0000001 * l1_norm.item()))

            if i % 10 == 0:
                net = net.eval()
                valx, valy= validate_labels(net, trainloaderfast)
                accuracy = metrics.balanced_accuracy_score(valy, valx)
                precision = metrics.precision_score(valy, valx)
                valx, valy = validate_logits(net, trainloaderfast)
                logits = [elt[0] for elt in valx]
                true = [elt[0] for elt in valy]
                auc = metrics.roc_auc_score(true, logits)
                print("Epoch {}: {}, {}, {}, {}".format(i,np.mean(losses),accuracy, precision))
                valx, valy = validate_labels(net, testloader)
                #print(metrics.roc_auc_score(valy, valx))
                accuracy = metrics.balanced_accuracy_score(valy, valx)
                precision = metrics.precision_score(valy, valx)
                valx, valy = validate_logits(net, testloader)
                logits = [elt[0] for elt in valx]
                true = [elt[0] for elt in valy]
                auc = metrics.roc_auc_score(true, logits)
                print("Epoch {}: {}, {}, {}, {}".format(i,np.mean(losses),accuracy, precision))
                if auc>best_val:
                    torch.save(net.state_dict(), outpath+"/"+str(model).split("(")[0]+"/best.pt")
                    best_val=auc
                    