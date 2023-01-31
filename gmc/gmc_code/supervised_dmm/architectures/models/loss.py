"""
Loss functions.
"""

from typing import Callable, Dict

import torch as th
from torch import nn
from torch.nn import functional as F


def cosine_sim(visual_emb: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
    """
    Calculate cosine similarity.

    Args:
        visual_emb: Visual embedding with shape (num_datapoints, dim_embedding)
        text_emb: Text embedding with shape (num_datapoints, dim_embedding)

    Returns:
        Cosine similariies with shape (num_datapoints, num_datapoints)
    """
    return visual_emb.mm(text_emb.t())


class ContrastiveLoss(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """
    def __init__(self, margin: float, max_violation: bool = False, in_norm: int = 1, out_norm: bool = True, use_cuda: bool = True):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.in_norm = in_norm
        self.out_norm = out_norm
        self.max_violation = max_violation
        self.use_cuda = use_cuda

    def forward(self, im, s):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        if self.in_norm:
            im_norm = F.normalize(im)
            s_norm = F.normalize(s)
        else:
            im_norm = im
            s_norm = s
        # compute image-sentence score matrix - how close is im(y) to s(x)
        scores = self.sim(im_norm, s_norm)                     #! positive and negative score (cos sim)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)                        #! positive score (cos sim)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)      #! margin + neg sim - pos sim
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals, where there is just the margin left
        mask: th.Tensor = th.eye(scores.shape[0]).bool()
        if self.use_cuda:
            mask = mask.cuda(non_blocking=True)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.out_norm:
            return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])
        return cost_s.sum() + cost_im.sum()