import torch
import random

from torch import nn

from .BaseLoss import BaseLoss


class MinTripletLossDML(BaseLoss):
    def __init__(self, margin: float = 1, regularizers: list = []):
        """Class of Triplet Loss taking minimum negative sample."""
        super().__init__(regularizers)
        self.margin = margin
        self.ReLU = nn.ReLU()

    def main(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """Method of forwarding main loss

        Args:
            embeddings_dict (dict): A dictionary of embddings which has following key and values.
                user_embedding : embeddings of user, size (n_batch, 1, d)
                pos_item_embedding : embeddings of positive item, size (n_batch, 1, d)
                neg_item_embedding : embeddings of negative item, size (n_batch, n_neg_samples, d)

            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.

        Return:
            torch.Tensor: loss,  L = Σ [m + pos_dist^2 - min(neg_dist)^2]
        """
        # added
        lamb = random.random()
        while lamb > 0.5:
            lamb = random.random()

        self.margin = self.margin * lamb
        user_embedding = embeddings_dict["user_embedding"]
        flipeed_user_embedding = torch.flip(user_embedding, dims=[0])
        mixed_user_embedding = lamb * user_embedding + (1-lamb) * flipeed_user_embedding
        neg_embeddings = embeddings_dict["neg_item_embedding"]
        # added end
        pos_dist = torch.cdist(
            mixed_user_embedding, embeddings_dict["pos_item_embedding"]
        )
        neg_dist = torch.cdist(
            mixed_user_embedding, neg_embeddings
        )
        min_neg_dist = torch.min(neg_dist, axis=2)
        pairwiseloss = self.ReLU(self.margin + pos_dist - min_neg_dist.values)

        loss = torch.mean(pairwiseloss)

        return loss
