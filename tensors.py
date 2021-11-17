# *********************
# Flattening samples
# *********************
randm_img = torch.randn(100,3,224,224)
# flatten all samples
randm_img = randm_img.view(100,-1)

print(randm_img.size())


# *********************
# One-hot encoding
# *********************
import torch
import torch.nn.functional as F

x = torch.tensor([4, 3, 2, 1, 0]) # list of targets
F.one_hot(x, num_classes=6)