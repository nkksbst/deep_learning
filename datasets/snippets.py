import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# *************************************************
# TYPICAL TRANSFORM
# *************************************************
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p = 1.0),
    transforms.RandomRotation(30),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.20, 0.224, 0.225]),
    transforms.ToTensor()
])

# *************************************************
# LOADING TORCH DATASETS
# *************************************************
train_data = datasets.MNIST(root = './data', train = True,
                            download = False, transform = transform)
# data is shuffled every epoch
loader = DataLoader(train_data, batch_size = 100, shuffle = True)
# *_data is a list of tuples (img_tensor, label)
# *_data[idx] is one sample

# show and save a sample image
image = train_data[0][0]
plt.imshow(image.reshape((28,28)), cmap = 'gray')
plt.savefig('/deep_learning/mnist_sample.png')

# *************************************************
# CUSTOM DATASET
# *************************************************
with Image.open(args.img_path)as im:
    plt.imshow(im)

# deciding standard image size
img_paths = glob.glob(args.data_dir + '*.file_extension')
rejected = []
img_sizes = []
for item in img_paths:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)

df = pd.DataFrame(img_sizes)
# mean and median of these sizes can be used now for standardizing image sizes
df[0].describe()
df[1].describe()


# *************************************************
# CUSTOM DATASET
# # getting the std and mean for transforms' normalization
# https://pytorch.org/vision/stable/models.html
# *************************************************
import torch
from torchvision import datasets, transforms as T

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
dataset = datasets.ImageNet(".", split="train", transform=transform)

means = []
stds = []
for img in subset(dataset):
    means.append(torch.mean(img))
    stds.append(torch.std(img))

mean = torch.mean(torch.tensor(means))
std = torch.mean(torch.tensor(stds))

# *************************************************
# Unnormalize
# *************************************************

mean = torch.tensor([0.4915, 0.4823, 0.4468])
std = torch.tensor([0.2470, 0.2435, 0.2616])

normalize = transforms.Normalize(mean.tolist(), std.tolist()) 

unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
img_unn = unnormalize(img)

plt.imshow(img_unn.permute(1, 2, 0))
plt.show()

# *************************************************
# Redisplay image obtained from dataloader
# *************************************************

# given a tensor from data_loader img

img = data_loader[i][0]
plt.imshow(np.transpose(im.numpy(), (1,2,0)))