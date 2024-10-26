import torch
import torchvision
import numpy as np

from sklearn.model_selection import train_test_split

import logging


class DataPreProcessor:
    def __init__(self):
        self.__logger = logging.getLogger('main')
        
        self.data_mean, self.data_std = self.__data_mean_std()
        
        self.__trainset_original = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=None)
        self.__testset = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=None)
        
        self.__logger.debug('data preprocessor init')
        
    def __data_mean_std(self):
        data = torchvision.datasets.CIFAR100("./data", train=True, download=True)
        x = np.concatenate([np.asarray(data[i][0]) for i in range(len(data))])

        mean = np.mean(x, axis=(0, 1))/255
        std = np.std(x, axis=(0, 1))/255

        mean = mean.tolist()
        std = std.tolist()
        self.__logger.debug(f'data mean: {mean}\tdata std: {std}')
        return mean, std
    
    def transform_data(self, train_original, test):
        self.__trainset_original.transform = train_original
        self.__testset.transform = test
        self.__logger.debug(f'set transform')
        
        
    def split_data(self, p=0.2):
        targets = self.__trainset_original.targets

        train_idx, val_idx = [], []
        for cls in range(100):
            cls_idx = [i for i, t in enumerate(targets) if t == cls]
            cls_train_idx, cls_val_idx = train_test_split(cls_idx, test_size=p, random_state=42)
            train_idx.extend(cls_train_idx)
            val_idx.extend(cls_val_idx)

        self.__train_dataset = torch.utils.data.Subset(self.__trainset_original, train_idx)
        self.__val_dataset = torch.utils.data.Subset(self.__trainset_original, val_idx)

        self.__logger.debug(f'data split - train size: {len(self.__train_dataset)}\tvalidation size: {len(self.__val_dataset)}')

    def get_data_loader(self, batch_size, num_worker=0):
        trainloader = torch.utils.data.DataLoader(self.__train_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
        validationloader = torch.utils.data.DataLoader(self.__val_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
        testloader = torch.utils.data.DataLoader(self.__testset, batch_size*2,pin_memory=True, num_workers=0)
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.testloader = testloader
