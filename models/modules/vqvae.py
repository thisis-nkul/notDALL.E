import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    FYI, GitHub Co-Pilot wrote a part this `-`
    """
    def __init__(self, num_embeddings, embedding_dim, k,
                 init_embeddings=None, commitment_cost=0.25,
                 embedding_scale=0.1):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.k = k
        self.commitment_cost = commitment_cost
        self.embedding_scale = embedding_scale

        if init_embeddings is not None:
            self.embeddings = nn.Parameter(init_embeddings)
        else:
            self.embeddings = nn.Embedding(num_embeddings, embedding_dim,
                                           padding_idx=0)
            # truncated normal
            self.embeddings.data.uniform_(-self.embedding_scale,
                                          self.embedding_scale)

    def forward(self, x):
        """
        :param x: (batch_size, embedding_dim)
        """
        batch_size = x.size(0)
        x_emb = self.embeddings(x)
        x_emb = x_emb.view(batch_size, -1)

        # Compute distances
        distances = (x_emb ** 2).sum(dim=1, keepdim=True) \
            + (self.embeddings.weight ** 2).sum(dim=1) \
            - 2 * x_emb.mm(self.embeddings.weight.transpose(0, 1))

        # Assignments
        assignments = distances.min(dim=1)[1]

        # Quantize
        quantized = self.embeddings.weight[assignments]

        # TODO: apply stop gradient here
        diff = quantized - x_emb
        loss = diff.pow(2).sum(dim=1).mean()

        # Add regularization
        loss += self.commitment_cost * self.embeddings.pow(2).sum()

        return loss, quantized, diff
