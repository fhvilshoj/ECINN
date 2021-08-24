import os
import torch
import configparser
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets

from .fakemnist import FakeMNIST

# Import from the IB-INN submodule
from data import Augmentor, LabelAugmentor

def celeb_dir():     
    if os.path.isfile('config.ini'): 
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        return cfg['data']['celeba_dir']
    else: 
        return './data'

# # # Modified version of https://github.com/VLL-HD/IB-INN/blob/master/data.py
# Added FakeMNIST 
# Added CelebA
# Always use IB-INN model structure.

class Dataset():
    def __init__(self, args):
        self.dataset      = args['data']['dataset']
        self.batch_size   = eval(args['data']['batch_size'])
        tanh              = eval(args['data']['tanh_augmentation'])
        self.sigma        = eval(args['data']['noise_amplitde'])
        unif              = eval(args['data']['dequantize_uniform'])
        label_smoothing   = eval(args['data']['label_smoothing'])
        channel_pad       = eval(args['data']['pad_noise_channels'])
        channel_pad_sigma = eval(args['data']['pad_noise_std'])
        if self.dataset == 'celeba':
            self.label = eval(args['data']['celeb_label'])

        if self.dataset in ['MNIST', 'fake_MNIST']: 
            beta = 0.5
            gamma = 2.
        else:
            beta = torch.Tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
            gamma = 1. / torch.Tensor((0.247, 0.243, 0.261)).view(-1, 1, 1)

        self.train_augmentor = Augmentor(False, self.sigma, unif, beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.test_augmentor =  Augmentor(True,  0.,         unif, beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.transform = T.Compose([T.ToTensor(), self.test_augmentor])

        if 'mnist' in self.dataset.lower():
            self.dims = (28, 28)
            if channel_pad:
                raise ValueError('needs to be fixed, channel padding does not work with mnist')
            self.channels = 1
            self.n_classes = 10
            self.label_mapping = list(range(self.n_classes))
            self.label_augment = LabelAugmentor(self.label_mapping)
            if self.dataset == 'MNIST': 
                data_dir = 'data/mnist_data'
                self.test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                                       transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                                       target_transform=self.label_augment)
                self.train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                                        transform=T.Compose([T.ToTensor(), self.train_augmentor]),
                                                        target_transform=self.label_augment)

            else: # fake MNIST
                data_dir = 'data/fake_mnist_data'
                self.test_data = FakeMNIST(data_dir, train=False, download=True,
                                           transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                           target_transform=self.label_augment)
                self.train_data = FakeMNIST(data_dir, train=True, download=True,
                                            transform=T.Compose([T.ToTensor(), self.train_augmentor]),
                                            target_transform=self.label_augment)

        elif self.dataset in ['celeba']:
            """
            | label | title               | dist.    |
            |     2 | Attractive          | 51.2505  |
            |    21 | Mouth_Slightly_Open | 48.3428  |
            |    31 | Smiling             | 48.208   |
            |    36 | Wearing_Lipstick    | 47.2436  |
            |    19 | High_Cheekbones     | 45.5032  |
            """
            self.n_classes     = 2
            self.label_mapping = list(range(self.n_classes))
            self.label_augment = LabelAugmentor(self.label_mapping)

            resolution      = args.getint('data', 'resolution', fallback=64)
            self.resolution = resolution
            self.dims       = (3 + channel_pad, self.resolution, self.resolution)
            self.channels   = 3 + channel_pad

            print("Appending: %s" % celeb_dir())
            sys.path.append(celeb_dir()) # root folder
            print("Resolution:", self.resolution) 

            from celeba.celeba_dataset import CelebADatasetSplit

            self.test_data = CelebADatasetSplit(resolution=self.resolution, 
                        mode='test', 
                        label=self.label, 
                        transform=T.Compose([T.ToTensor(), self.test_augmentor]), 
                        target_transform=self.label_augment
                    )
            self.train_data = CelebADatasetSplit(resolution=self.resolution, 
                        mode='train', 
                        label=self.label, 
                        # transform=T.Compose([T.ToTensor(), self.test_augmentor]), 
                        transform=T.Compose([T.RandomHorizontalFlip(),
                                             # T.ColorJitter(0.1, 0.1, 0.05),
                                             T.ToTensor(),
                                             self.train_augmentor]),
                        target_transform=self.label_augment
                    )

        elif self.dataset in ['CIFAR10', 'CIFAR100']:
            self.dims = (3 + channel_pad, 32, 32)
            self.channels = 3 + channel_pad

            if self.dataset == 'CIFAR10':
                data_dir = 'data/cifar_data'
                self.n_classes = 10
                dataset_class = torchvision.datasets.CIFAR10
            else:
                data_dir = 'data/cifar100_data'
                self.n_classes = 100
                dataset_class = torchvision.datasets.CIFAR100

            self.label_mapping = list(range(self.n_classes))
            self.label_augment = LabelAugmentor(self.label_mapping)

            self.test_data = dataset_class(data_dir, train=False, download=True,
                                                   transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                                   target_transform=self.label_augment)
            self.train_data = dataset_class(data_dir, train=True, download=True,
                                                   transform=T.Compose([T.RandomHorizontalFlip(),
                                                                       T.ColorJitter(0.1, 0.1, 0.05),
                                                                       T.Pad(8, padding_mode='edge'),
                                                                       T.RandomRotation(12),
                                                                       T.CenterCrop(36),
                                                                       T.RandomCrop(32),
                                                                       T.ToTensor(),
                                                                       self.train_augmentor]),
                                                    target_transform=self.label_augment)

        else:
            raise ValueError(f"what is this dataset, {args['data']['dataset']}?")

        val_size = 100
        self.train_data, self.val_data = torch.utils.data.random_split(self.train_data, (len(self.train_data) - val_size, val_size))

        self.val_x = torch.stack([x[0] for x in self.val_data], dim=0).cuda()
        self.val_y = self.onehot(torch.LongTensor([x[1] for x in self.val_data]).cuda(), label_smoothing)

        self.train_loader  = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                   num_workers=6, pin_memory=True, drop_last=True)
        self.test_loader   = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True, drop_last=False)

    def show_data_hist(self):
        x = self.val_x.cpu().numpy()
        plt.hist(x.flatten(), bins=200)
        plt.show()

    def de_augment(self, x):
        return self.test_augmentor.de_augment(x)

    def augment(self, x):
        return self.test_augmentor(x)

    def onehot(self, l, label_smooth=0):
        y = torch.cuda.FloatTensor(l.shape[0], self.n_classes).zero_()
        y.scatter_(1, l.view(-1, 1), 1.)
        if label_smooth:
            y = y * (1 - label_smooth) + label_smooth / self.n_classes
        return y



    labels = {
            'MNIST':        {i: str(i) for i in range(10)},
            'fake_MNIST':   {i: str(i) for i in range(10)},
            'CIFAR10':      {i: s for i, s in enumerate(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])},
            'celeba':       {2: "Attractive", 21: "Mouth_Slightly_Open", 31: "Smiling", 36: "Wearing_Lipstick", 19: "High_Cheekbones"}
    }
            
    def get_labels(self, idxs):

        if isinstance(idxs, torch.Tensor):
            if len(idxs.shape)==2 and idxs.shape[1] == 10: # one-hot
                idxs = idxs.max(1, False)[1] # convert to index
            idxs = idxs.tolist()

        d = Dataset.labels.get(self.dataset, ['Unknown']*len(idxs)) 
        
        l = []
        if self.dataset == 'celeba': 
            suffix = d[self.label]
            for i in idxs:
                p = "" if i == 1 else "not"
                l.append("%s %s" % (p, suffix))
        else: 
            for i in idxs:
                l.append(d[i])
        return l
