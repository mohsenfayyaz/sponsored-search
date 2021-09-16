import torch

from src.RepresentationBuilder import RepresentationBuilder


class QueryAdCoordinator(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(QueryAdCoordinator, self).__init__()
        self.representation_builder = RepresentationBuilder(vocab_size, embedding_dim)
        self.loss = torch.nn.CosineEmbeddingLoss(margin=0)

    def forward(self, x1, x2, attention_mask1, attention_mask2):
        x1_repr = self.representation_builder(x1, attention_mask1)
        x2_repr = self.representation_builder(x2, attention_mask2)
        print(x1_repr)
        print(x2_repr)
        print(self.loss(x1_repr, x2_repr, target=torch.tensor([-1, 1])))
