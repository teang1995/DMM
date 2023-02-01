import torch
import random
from torch import nn

from .BaseLoss import BaseLoss


class BprLoss(BaseLoss):
    """Class of Triplet Loss taking sum of negative sample."""

    def __init__(self, margin: float = 1, regularizers: list = []):
        super().__init__(regularizers)
        self.margin = margin
        self.ReLU = nn.ReLU()

    def main(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """Method of forwarding main loss

        Args:
            embeddings_dict (dict): A dictionary of embddings which has following key and values
                user_embedding : embeddings of user, size (n_batch, 1, d)
                pos_item_embedding : embeddings of positive item, size (n_batch, 1, d)
                neg_item_embedding : embeddings of negative item, size (n_batch, 1, d)

            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.

        Return:
            torch.Tensor : loss, L = Σ [m + pos_dist^2 - min(neg_dist)^2]
        """
        user_emb = embeddings_dict['user_embedding']
        pos_emb = embeddings_dict['pos_item_embedding']
        neg_emb = embeddings_dict['neg_item_embedding']
        
        pos_scores = torch.sum((user_emb * pos_emb).squeeze(1), dim=1)
        neg_scores = torch.sum((user_emb * neg_emb).squeeze(1), dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        return loss
