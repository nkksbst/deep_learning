import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import sys
sys.path.insert(0, '/deep_learning')
import argparse
from utils import conv_output_shape
import datasets.datasets as datasets

class SomeModel(nn.Module):
    pass

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--some_arg", type = int, default = 64)
    args = parser.parse_args()

    torch.manual_seed(101)
    model = SomeModel((28,28), input_channel = 1, output_size = 10, channels = [6, 16])
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
                y_val = model(x_test)

                predicted = torch.max(y_val, dim = 1)[1]
                tst_corr += (predicted == y_test).sum()
                

    train_losses.append(loss)
    train_correct.append(trn_corr)

if __name__ == '__main__':
    main()