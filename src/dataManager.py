'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-10 20:26:10
LastEditors: LetMeFly
LastEditTime: 2024-07-11 09:37:00
FilePath: /master/src/datamanager.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import List
import random

# 数据管理类
class DataManager:
    def __init__(self, num_clients: int, batch_size: int, datasize_perclient: int, datasize_valide: int):
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.datasize_perclient = datasize_perclient
        self.datasize_valide = datasize_valide
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        
    def get_clients_data_loaders(self) -> List[DataLoader]:
        dataset_size = len(self.train_dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        clients_data_loaders = []
        start_idx = 0
        for _ in range(self.num_clients):
            split_indices = indices[start_idx:start_idx + self.datasize_perclient]
            subset = Subset(self.train_dataset, split_indices)
            data_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            clients_data_loaders.append(data_loader)
            start_idx += self.datasize_perclient
        
        return clients_data_loaders

    def get_val_loader(self) -> DataLoader:
        test_indices = list(range(len(self.test_dataset)))
        random.shuffle(test_indices)
        val_indices = test_indices[:self.datasize_valide]
        val_subset = Subset(self.test_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        return val_loader
