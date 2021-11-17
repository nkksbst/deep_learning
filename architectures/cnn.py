import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import sys
sys.path.insert(0, '/home/deep_learning')
import argparse
from utils import conv_output_shape
import datasets.datasets as datasets

class CNN(nn.Module):
    def __init__(self, image_dim, input_channel = 1, output_size = 10, channels = [6, 16]):
        """
        image_dim: a tuple of (H, W)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = channels[0], kernel_size = 3, stride = 1)
        h, w = conv_output_shape(image_dim, kernel_size=3, stride=1, pad=0, dilation=1)
        self.conv2 = nn.Conv2d(in_channels = channels[0], out_channels = channels[1], kernel_size = 3, stride = 1)
        # h, w are halfed here because maxpool 2d is present in the forward func
        h, w = conv_output_shape((h/2,w/2), kernel_size=3, stride=1, pad=0, dilation=1)
        self.fc1 = nn.Linear(channels[1] * h * w, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.fc1(x.view(x.size()[0], -1))
        x = self.fc2(x)

        return F.log_softmax(x, dim = 1)

def train_one_epoch(train_loader):
    pass
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--eval_batch_size", type = int, default=128)
    parser.add_argument("--dataset", type = str)
    parser.add_argument("--epochs", type = int, default=1)

    args = parser.parse_args()

    torch.manual_seed(101)
    model = CNN((28,28), input_channel = 1, output_size = 10, channels = [6, 16])
    transform = transforms.ToTensor()
    train_loader, test_loader = datasets.get_loader(args, transform, download = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    trn_corr = 0
    train_losses = []
    train_correct = []
    for epoch in range(args.epochs):

        trn_corr = 0
        tst_corr = 0

        for b, (images, labels) in enumerate(train_loader):

            b += 1
            y_pred = model(images)
            loss = criterion(y_pred, labels)

            predicted = torch.max(y_pred, dim = 1)[1]
            batch_corr = (predicted == labels).sum()

            trn_corr += batch_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % args.batch_size * 10 == 0:
                acc = trn_corr.item() * 100 / (args.batch_size * b) 
                print(f'Epoch {epoch} batch {b} loss: {loss.item()} accuracy: {acc}')
        with torch.no_grad():
            for b, (x_test, y_test) in enumerate(test_loader):
                y_val = model(x_test, -1)

                predicted = torch.max(y_val, dim = 1)[1]
                tst_corr += (predicted == y_test).sum()

    train_losses.append(loss)
    train_correct.append(trn_corr)

if __name__ == '__main__':
    main()