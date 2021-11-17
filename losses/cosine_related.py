
import torch
import torch.nn as nn

# Cosine Similarity
# reference: https://discuss.pytorch.org/t/how-to-add-cosine-similarity-score-in-cross-entropy-loss/64401
x = torch.randn(100, 128)
x_ = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6) # eps: to avoid nan/division by zero
output = cos(x, x_)


# Cosine Embedding Loss
# gives choices whether the pair should be similar (y=1) or dissimilar (y=-1)
batch_size = 32
embedding_size = 32
x = torch.randn(batch_size, embedding_size)
x_ = torch.randn(batch_size, embedding_size)
y = torch.ones(batch_size)
emb_loss = nn.CosineEmbeddingLoss(reduction = 'mean') # reduction can also be sum or none
loss = emb_loss(x, x_, y)