import sys
sys.path.insert(0, '/home/deep_learning')
import argparse

import datasets.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from cnn import CNN


import numpy as np
from matplotlib import pyplot as plt

def train(args, model, optimizer, criterion, data_loaders):
    
    train_loader, test_loader = data_loaders

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
                b += 1
                y_val = model(x_test)

                predicted = torch.max(y_val, dim = 1)[1]
                tst_corr += (predicted == y_test).sum()
            test_acc = tst_corr.item() * 100 / (args.eval_batch_size * b) 
            print(f'Test accuracy: {test_acc}')

    train_losses.append(loss)
    train_correct.append(trn_corr) 

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'fashion_mnist')
    parser.add_argument('--data_dir', type = str, default = '/home/deep_learning/data')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--eval_batch_size', type = int, default = 64)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--epochs', type = int, default = 1)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader, test_loader = datasets.get_loader(args, transform, download = False)


    for _, (images, _) in enumerate(train_loader):
        break

    n_rows = 1
    n_cols = 10
    plt.title('Sample Training Images')

    fig = plt.figure()

    for i in range(1, n_rows * n_cols):
        image = images[i][0]
        fig.add_subplot(n_rows, n_cols, i)
        plt.imshow(image.numpy(), cmap = 'gray')

    plt.show()
    plt.savefig('sample_multi_img.png')   

    model = CNN((28,28), input_channel = 1, 
                            output_size = 10, 
                            channels = [6, 16])

    torch.manual_seed(101)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()

    train(args, model, optimizer, criterion, (train_loader, test_loader))
if __name__ == '__main__':
    main()