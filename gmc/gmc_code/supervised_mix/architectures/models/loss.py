import math
import torch
from torch import nn


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def clip_loss(logits,targets):
    img_loss = cross_entropy(logits,targets)
    txt_loss = cross_entropy(logits.T,targets)
    #txt_loss = cross_entropy(logits.T,targets.T)
    final_loss = ((img_loss + txt_loss)/2.0 ).mean()
    return final_loss


def clip_loss2(logits1,targets,logits2=None,targets2=None):
    loss = cross_entropy(logits1,targets).mean()
    
    if logits2 != None and targets2 != None :
        loss += cross_entropy(logits2,targets).mean()
    return loss


def calc_mix_loss(logits_mix,lamb,mode="mix",c=-1,l=1):
    matrix_sz = logits_mix.shape[0]
    if c>0 :
        #Compute 2D-geometric lamb
        lamb1 = math.sqrt(  lamb**2       + c**2    )
        lamb2 = math.sqrt(  (1-lamb)**2   + c**2    )
        #normalize
        lamb1,lamb2 = lamb1/(lamb1+lamb2) , lamb2/(lamb1+lamb2)
    else :
        lamb1 = lamb
        lamb2 = 1-lamb
    if mode == "mix":
        targets_mix = torch.eye(matrix_sz).to("cuda:0")*lamb1  +  (lamb2)*torch.flip(torch.eye(matrix_sz).to("cuda:0"),dims=[0])
    elif mode == "temp":
        targets_mix = torch.eye(matrix_sz).to("cuda:0")
        logits_mix  = logits_mix*lamb
    targets_mix = targets_mix.to("cuda:0")
    if l==1 :
        return clip_loss(logits_mix, targets_mix)
    if l==2 :
        return clip_loss2(logits_mix, targets_mix)
