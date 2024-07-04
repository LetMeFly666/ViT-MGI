以下是我当前代码

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
        self.train_indices = self._create_train_indices()
        
    def _create_train_indices(self) -> List[int]:
        dataset_size = len(self.train_dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        return indices

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

    def get_val_loader(self) -> DataLoader:
        test_indices = list(range(len(self.test_dataset)))
        random.shuffle(test_indices)
        val_indices = test_indices[:self.datasize_valide]
        val_subset = Subset(self.test_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        return val_loader
```

这个类中，`train_indices`是在`init`的时候就确定好的。后续无论再怎么随机，都是在`train_indices`中选择的数据。是吗？

是这样的话，就会导致有很多数据重来都没有被使用过。

请修改这个类（不要返回这个类之外的函数），使得每次调用`get_clients_data_loaders`时，得到的都是真正的随机的数据，而不是初始化时候的那些数据。