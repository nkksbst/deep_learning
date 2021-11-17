from torch.utils.data import DataLoader
from torchvision import datasets

def get_loader(args, transform, download):
    if args.dataset == 'mnist':

        train_data = datasets.MNIST(root = args.data_dir, train = True,
                                    download = download, transform = transform)
        test_data = datasets.MNIST(root = args.data_dir, train = True,
                                    download = download, transform = transform)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                                    shuffle = True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size = args.eval_batch_size, 
                                    shuffle = True, drop_last=True)

        return train_loader, test_loader
    elif args.dataset == 'fashion_mnist':
        train_data = datasets.FashionMNIST(root = args.data_dir, train = True,
                                    download = download, transform = transform)
        test_data = datasets.FashionMNIST(root = args.data_dir, train = True,
                                    download = download, transform = transform)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                                    shuffle = True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size = args.eval_batch_size, 
                                    shuffle = True, drop_last=True)

        return train_loader, test_loader       
    elif args.dataset == 'tinyimagenet':
        pass
    elif args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(root = args.data_dir, train = True,
                            download = False, transform = transform)

        test_data = datasets.CIFAR10(root = args.data_dir, train = True,
                                    download = False, transform = transform)
        
        train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                                    shuffle = True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size = args.eval_batch_size, 
                                    shuffle = True, drop_last=True)    
    
        return train_loader, test_loader

    elif args.dataset == 'cifar100':
        train_data = datasets.CIFAR10(root = args.data_dir, train = True,
                            download = False, transform = transform)

        test_data = datasets.CIFAR10(root = args.data_dir, train = True,
                                    download = False, transform = transform)
        
        train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                                    shuffle = True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size = args.eval_batch_size, 
                                    shuffle = True, drop_last=True)    
    
        return train_loader, test_loader

    elif args.dataset == 'custom-dataset':
        train_data = datasets.ImageFolder(root = args.data_dir,
                                            transform = transform)

    else:
        raise Exception('dataset %s not supported' % args.dataset)