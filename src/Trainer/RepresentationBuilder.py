from src.Trainer.SpanPooling import *


class RepresentationBuilder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super(RepresentationBuilder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.pooling_module = get_pooling_module("avg")

    def forward(self, input_ids, attention_mask):
        out = self.embedding(input_ids)
        out = self.pooling_module(out, attention_mask)
        return out
