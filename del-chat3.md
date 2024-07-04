下面代码中，`self.test_dataset`并未在后续使用到。下面代码中测试集也变成了从训练集中划分。

这样是不合理的。请修改下面代码，训练集每次就只作为训练来使用，而测试的时候每次从测试集中随机挑选`datasize_valide`个来测试。

```
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
        self.val_loader = self._create_val_loader()
        self.train_indices = self._create_train_indices()
        
    def _create_val_loader(self) -> DataLoader:
        dataset_size = len(self.train_dataset)
        assert(self.num_clients * self.datasize_perclient + self.datasize_valide <= dataset_size)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        val_indices = indices[:self.datasize_valide]
        val_subset = Subset(self.train_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        return val_loader
    
    def _create_train_indices(self) -> List[int]:
        dataset_size = len(self.train_dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[self.datasize_valide:]
        return train_indices

    def get_clients_data_loaders(self) -> List[DataLoader]:
        random.shuffle(self.train_indices)
        clients_data_loaders = []
        start_idx = 0
        for _ in range(self.num_clients):
            split_indices = self.train_indices[start_idx:start_idx + self.datasize_perclient]
            subset = Subset(self.train_dataset, split_indices)
            data_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            clients_data_loaders.append(data_loader)
            start_idx += self.datasize_perclient
        return clients_data_loaders
```