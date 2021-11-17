def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 64)

    args = parser.parse_args()

    # TODO set seed
    # torch.manual_seed(101)

    # TODO define model
    # model = somemodel()

    # TODO load datasets
    # transform = 
    # train_loader, test_loader = 

    # TODO define criterion and optimizer
    # criterion = 
    # optimizer = 

    trn_corr = 0
    train_losses = []
    train_correct = []
    
    for epoch in range(args.epochs):

        trn_corr = 0
        tst_corr = 0

        for b, (x, y) in enumerate(train_loader):

            b += 1
            y_pred = model(x)
            loss = criterion(y_pred, y)

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