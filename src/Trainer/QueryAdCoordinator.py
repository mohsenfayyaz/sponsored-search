import torch

from src.Trainer.RepresentationBuilder import RepresentationBuilder


class QueryAdCoordinator(torch.nn.Module):
    def __init__(self, query_vocab_size, ad_vocab_size, embedding_dim, learning_rate):
        super(QueryAdCoordinator, self).__init__()
        self.query_representation_builder = RepresentationBuilder(query_vocab_size, embedding_dim)
        self.ad_representation_builder = RepresentationBuilder(ad_vocab_size, embedding_dim)

        self.loss = torch.nn.CosineEmbeddingLoss(margin=0)
        self.optimizer = self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0)

    def forward(self, query, ad, query_attention_mask1, ad_attention_mask):
        query_repr = self.query_representation_builder(query, query_attention_mask1)
        ad_repr = self.ad_representation_builder(ad, ad_attention_mask)
        return query_repr, ad_repr

    def save_model(self, path="QueryAdCoordinator_checkpoint.pt"):
        # torch.save(self.state_dict(), path)
        torch.save(self, path)
        print(f"Model saved at {path}")

    def load_model(self, path="QueryAdCoordinator_checkpoint.pt"):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def build_ad_representation(self, ad_id_tensor, attention_mask):
        return self.ad_representation_builder(ad_id_tensor, attention_mask)

    def build_query_representation(self, query_tensor, attention_mask):
        return self.query_representation_builder(query_tensor, attention_mask)
