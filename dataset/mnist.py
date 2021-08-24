import torch
import numpy as np 
import os
import os.path
import torchvision
import torchvision 

class FakeMNIST(torchvision.datasets.MNIST):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=False, seed=42): 

        super(FakeMNIST, self).__init__(data_dir, train=train, transform=transform, target_transform=target_transform, download=download)

        if self._check_first():
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
 
            self.reorder_dataset()
            if self.train:
                data_file = self.training_file
            else:
                data_file = self.test_file
            self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def _check_first(self):
        return not os.path.exists(os.path.join(self.processed_folder, "done.txt"))

    def reorder_dataset(self): 
        print("Reordering")
        for data_file in [self.training_file, self.test_file]:
            data, targets = torch.load(os.path.join(self.processed_folder, data_file))

            perm = np.random.permutation(data.shape[0])
            data = data[perm]
            labels = torch.arange(10).long().view(10, 1).repeat(1,data.shape[0]//10)
            labels = labels.view(-1) # [0, 0, 0, ..., 1, 1, 1, ... , 9, 9, 9]
            data[torch.arange(data.shape[0]), labels, 0] = 255

            perm = np.random.permutation(data.shape[0])
            data = data[perm]
            labels = labels[perm]

            with open(os.path.join(self.processed_folder, data_file), 'wb') as f:
                torch.save((data, labels), f)
        with open(os.path.join(self.processed_folder, "done.txt"), 'w') as f:
            f.write("Done")

# # # Modified version of https://github.com/VLL-HD/IB-INN/blob/master/data.py
# Added FakeMNIST 
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

        if self.dataset in ['MNIST', 'fake_MNIST']: 
            beta = 0.5
            gamma = 2.
        else:
            beta = torch.Tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
            gamma = 1. / torch.Tensor((0.247, 0.243, 0.261)).view(-1, 1, 1)

        self.train_augmentor = Augmentor(False, self.sigma, unif, beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.test_augmentor =  Augmentor(True,  0.,         unif, beta, gamma, tanh, channel_pad, channel_pad_sigma)
        self.transform = T.Compose([T.ToTensor(), self.test_augmentor])

        if self.dataset in ['MNIST', 'fake_MNIST']:
            self.dims = (28, 28)
            if channel_pad:
                raise ValueError('needs to be fixed, channel padding does not work with mnist')
            self.channels = 1
            self.n_classes = 10
            self.label_mapping = list(range(self.n_classes))
            self.label_augment = LabelAugmentor(self.label_mapping)
            if self.dataset == 'MNIST': 
                data_dir = 'mnist_data'
                self.test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                                       transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                                       target_transform=self.label_augment)
                self.train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                                        transform=T.Compose([T.ToTensor(), self.train_augmentor]),
                                                        target_transform=self.label_augment)
            else: # fake MNIST
                data_dir = 'fake_mnist_data'
                self.test_data = FakeMNIST(data_dir, train=False, download=True,
                                           transform=T.Compose([T.ToTensor(), self.test_augmentor]),
                                           target_transform=self.label_augment)
                self.train_data = FakeMNIST(data_dir, train=True, download=True,
                                            transform=T.Compose([T.ToTensor(), self.train_augmentor]),
                                            target_transform=self.label_augment)


        elif self.dataset in ['CIFAR10', 'CIFAR100']:
            self.dims = (3 + channel_pad, 32, 32)
            self.channels = 3 + channel_pad

            if self.dataset == 'CIFAR10':
                data_dir = 'cifar_data'
                self.n_classes = 10
                dataset_class = torchvision.datasets.CIFAR10
            else:
                data_dir = 'cifar100_data'
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

        self.train_data, self.val_data = torch.utils.data.random_split(self.train_data, (len(self.train_data) - 1024, 1024))

        self.val_x = torch.stack([x[0] for x in self.val_data], dim=0).cuda()
        self.val_y = self.onehot(torch.LongTensor([x[1] for x in self.val_data]).cuda(), label_smoothing)

        self.train_loader  = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                   num_workers=6, pin_memory=True, drop_last=True)
        self.test_loader   = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True, drop_last=True)

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
