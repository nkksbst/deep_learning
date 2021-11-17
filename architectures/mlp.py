import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

import argparse
import sys


sys.path.insert(0, '/deep_learning')
import datasets.datasets as datasets

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size = 784, output_size = 10, layers = [120, 84]):
        
        super().__init__()

        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim = 1)


def train_one_epoch(train_loader):
    pass
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--eval_batch_size", type = int, default=128)
    parser.add_argument("--dataset", type = str)
    parser.add_argument("--epochs", type = int, default=10)

    args = parser.parse_args()

    torch.manual_seed(101)
    model = MultilayerPerceptron()
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
            y_pred = model(images.view(args.batch_size, -1))
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
                y_val = model(x_test.view(x_test.size()[0], -1))

                predicted = torch.max(y_val, dim = 1)[1]
                tst_corr += (predicted == y_test).sum()

    train_losses.append(loss)
    train_correct.append(trn_corr)

if __name__ == '__main__':
    main()