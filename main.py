'''
Author: LetMeFly
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-04 11:20:11
'''
import datetime
getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
now = getNow()
# del datetime
from src.utils import initPrint, TimeRecorder, Ploter
from typing import List, Optional, Tuple

initPrint(now)
print(now)

timeRecorder = TimeRecorder()
timeRecorder.addRecord('Start', getNow())


    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import ViTForImageClassification, ViTConfig
import numpy as np
import random
import copy


# 参数
num_clients = 5
batch_size = 32
num_rounds = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(f'./result/{now}/.env', 'w') as f:
    f.write(f'num_clients = {num_clients}\nbatch_size = {batch_size}\nnum_rounds = {num_rounds}\ndevice = {device}\n')

# 定义ViT模型
class ViTModel(nn.Module):
    def __init__(self, num_classes=10, device: str=device, name: str=None):
        super(ViTModel, self).__init__()
        model_path = './data/vit_base_patch16_224'
        config = ViTConfig.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path, config=config)
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.model.to(device)  # 移动模型到设备
        self.name = 'defaultName'
    
    def forward(self, x):
        return self.model(x).logits

    def setName(self, name: str) -> None:
        self.name = name
    
    def getName(self) -> str:
        return self.name

# 客户端类
class Client:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model: Optional[ViTModel] = None
        # self.global_model: Optional[ViTModel] = None
    
    def set_model(self, model: ViTModel, device: str, name: str=None):
        self.model = model
        self.model.to(device)
        self.model.setName(name)
        # self.global_model = copy.deepcopy(self.global_model)
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
    
    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str):
        self.model.to(device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        total_loss = 0.0
        batchNum = 0
        for images, labels in self.data_loader:
            batchNum += 1
            optimizer.zero_grad()  # 每个批次前清零梯度
            images, labels = images.to(device), labels.to(device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # 计算当前批次的梯度
            thisLoss = loss.item()
            total_loss += thisLoss
            optimizer.step()
            # print(f'{self.getName()} batch{batchNum} loss: {thisLoss}')
        # 计算梯度变化
        final_state_dict = self.model.state_dict()
        gradient_changes = {}
        for key in self.initial_state_dict:
            gradient_changes[key] = final_state_dict[key] - self.initial_state_dict[key]
        return gradient_changes, total_loss / len(self.data_loader)

    def getName(self) -> str:
        return self.model.getName()

# 服务器类
class Server:
    def __init__(self, model: ViTModel, device: str):
        self.global_model = model
        self.global_model.to(device)
        self.device = device
    
    def distribute_model(self, clients: List[Client]):
        for th, client in enumerate(clients):
            client.set_model(copy.deepcopy(self.global_model), device=self.device, name=f'Client{th + 1}')
    
    def aggregate_gradients(self, grads_list):
        avg_grads = {}
        for grads in grads_list:
            for key in grads:
                if key not in avg_grads:
                    avg_grads[key] = grads[key].clone() / len(grads_list)
                else:
                    avg_grads[key] += grads[key] / len(grads_list)
        return avg_grads
    
    def update_model(self, gradient_changes):
        # 获取全局模型的状态字典
        global_state_dict = self.global_model.state_dict()
        # 更新全局模型的参数
        for key in global_state_dict:
            global_state_dict[key] += gradient_changes[key]
        # 将更新后的状态字典加载回全局模型
        self.global_model.load_state_dict(global_state_dict)

    def evaluate(self, data_loader, device: str):
        self.global_model.to(device)
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

def get_data_loaders(num_clients, batch_size) -> Tuple[List[Client], DataLoader]:
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
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
    
    # 为验证集创建数据加载器
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return clients, val_loader

# 获取数据加载器
clients, val_loader = get_data_loaders(num_clients, batch_size)

# 初始化服务器
global_model = ViTModel(device=device, name='GlobalModel')
server = Server(global_model, device)

# 联邦学习过程
criterion = nn.CrossEntropyLoss()

ploter = Ploter(x='batch', y=['loss', 'accuracy'], title='loss and accuracy', filename=f'./result/{now}/lossAndAccuracy.svg')
accuracy = server.evaluate(val_loader, device)
timeRecorder.addRecord(f'init accuracy: {accuracy*100:.2f}%')
import math  # TODO: 计算真正的loss
ploter.addData(x=0, y={'loss': math.nan, 'accuracy': accuracy})

for round_num in range(num_rounds):
    timeRecorder.addRecord(f'Round {round_num + 1} of {num_rounds}')
    # 分发当前的全局模型给所有客户端
    server.distribute_model(clients)
    # 每个客户端计算梯度
    grads_list = []
    total_loss = 0.0
    for th, client in enumerate(clients):
        timeRecorder.addRecord(f'Round {round_num + 1}/{num_rounds} client {th}/{num_clients} is computing gradients...')
        grads, loss = client.compute_gradient(criterion, device)
        print(f'Client {th + 1} has computed gradients. | {getNow()}')
        grads_list.append(grads)
        total_loss += loss
    avg_loss = total_loss / num_clients
    print(f"Average loss: {avg_loss} | {getNow()}")
    
    # 服务器聚合梯度并更新全局模型
    avg_grads = server.aggregate_gradients(grads_list)
    server.update_model(avg_grads)
    
    # 计算在验证集上的准确率
    print(f'Begin to evaluate accuracy... | {getNow()}')
    total_accuracy = 0.0
    accuracy = server.evaluate(val_loader, device)
    timeRecorder.addRecord(f"Client {th}\'s accuracy: {accuracy*100:.2f}%")
    ploter.addData(x=round_num + 1, y={'loss': avg_loss, 'accuracy': accuracy})

print("Federated learning completed.")
timeRecorder.printAll()
