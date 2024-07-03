'''
Author: LetMeFly
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-03 11:16:31
'''
import datetime
getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
now = getNow()
# del datetime
from src.utils import initPrint
from typing import List, Optional

initPrint(now)
print(now)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import ViTForImageClassification, ViTConfig
import numpy as np
import random


# 定义ViT模型
class ViTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTModel, self).__init__()
        model_path = './data/vit_base_patch16_224'
        config = ViTConfig.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path, config=config)
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
    
    def forward(self, x):
        return self.model(x).logits

# 客户端类
class Client:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model: Optional[ViTModel] = None
    
    def set_model(self, model):
        self.model = model
    
    def compute_gradient(self, criterion, device):
        self.model.to(device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        optimizer.zero_grad()
        
        total_loss = 0.0
        for images, labels in self.data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
        
        grads = []
        for param in self.model.parameters():
            grads.append(param.grad.clone().cpu())
        
        return grads, total_loss / len(self.data_loader)

# 服务器类
class Server:
    def __init__(self, model, device):
        self.global_model = model
        self.device = device
    
    def distribute_model(self, clients: List[Client]):
        for client in clients:
            client.set_model(self.global_model)
    
    def aggregate_gradients(self, grads_list):
        avg_grads = []
        for grads in zip(*grads_list):
            avg_grads.append(sum(grads) / len(grads))
        return avg_grads
    
    def update_model(self, grads):
        for param, grad in zip(self.global_model.parameters(), grads):
            param.grad = grad.to(self.device)
        optimizer = optim.SGD(self.global_model.parameters(), lr=0.01, momentum=0.9)
        optimizer.step()

def get_data_loaders(num_clients, batch_size) -> List[Client]:
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 将数据集划分给多个客户端
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    
    clients = []
    for split in split_indices:
        subset = Subset(train_dataset, split)
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        clients.append(Client(data_loader))
    
    return clients

# 参数
num_clients = 5
batch_size = 32
num_rounds = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据加载器
clients = get_data_loaders(num_clients, batch_size)

# 初始化服务器
global_model = ViTModel()
server = Server(global_model, device)

# 联邦学习过程
criterion = nn.CrossEntropyLoss()


class TimeRecorder():
    def __init__(self, eventName: str, eventTime: str):
        self.eventName = eventName
        self.eventTime = eventTime
    
    def __str__(self):
        return f'{self.eventName} - {self.eventTime}'
    

timeList = []

def printTimeList():
    toPrint = 'TimeList:'
    for i in range(len(timeList)):
        toPrint += f'\n{i:02d}: {timeList[i]}'
    print(toPrint)


for round_num in range(num_rounds):
    print(f"Round {round_num+1} of {num_rounds} | {getNow()}")
    
    # 分发当前的全局模型给所有客户端
    server.distribute_model(clients)
    
    # 每个客户端计算梯度
    grads_list = []
    total_loss = 0.0
    for th, client in enumerate(clients):
        print(f'Client {th} is computing gradients... | {getNow()}')
        grads, loss = client.compute_gradient(criterion, device)
        print(f'Client {th} has computed gradients. | {getNow()}')
        grads_list.append(grads)
        total_loss += loss
    
    avg_loss = total_loss / num_clients
    print(f"Average loss: {avg_loss} | {getNow()}")
    
    # 服务器聚合梯度并更新全局模型
    avg_grads = server.aggregate_gradients(grads_list)
    server.update_model(avg_grads)

print("Federated learning completed.")

