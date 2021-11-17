"""
more info here:
https://pytorch.org/vision/stable/models.html
these are off-the-shell models from pytorch
nice to work with when finetuning on downstream tasks
"""
import torchvision.models as models
import torch.nn as nn
import torch

alexnet = models.alexnet(pretrained = True)

for param in alexnet.parameters():
    param.requires_grad = False

torch.manual_seed(42)

alexnet.classifier = nn.Sequential(
    nn.Linear(9126, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 2),
    nn.LogSoftmax(dim = 1)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.classifier.parameters, lr = 0.001)

