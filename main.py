'''
Author: LetMeFly
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-05 21:26:09
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
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import ViTForImageClassification, ViTConfig
import numpy as np
import random
import copy
from sklearn.decomposition import PCA


# 参数/配置
num_clients = 10          # 客户端数量
batch_size = 32           # 每批次多少张图片
num_rounds = 32           # 总轮次
epoch_client = 1          # 每个客户端的轮次
datasize_perclient = 32   # 每个客户端的数据量
datasize_valide = 1000    # 测试集大小
learning_rate = 0.001     # 步长
ifPCA = True              # 是否启用PCA评价 
ifCleanAnoma = True       # 是否清理PCA抓出的异常数据
PCA_rate = 1              # PCA偏离倍数
attackList = [0, 1, 2]    # 恶意客户端下标
attack_rate = 1           # 攻击强度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(f'./result/{now}/config.env', 'w') as f:
    f.write(f"""num_clients = {num_clients}
batch_size = {batch_size}
num_rounds = {num_rounds}
epoch_client = {epoch_client}
datasize_perclient = {datasize_perclient}
datasize_valide = {datasize_valide}
learning_rate = {learning_rate}
ifPCA = {ifPCA}
ifCleanAnoma = {ifCleanAnoma}
PCA_rate = {PCA_rate}
attackList = {attackList}
attack_rate = {attack_rate}
device = {device}
""")

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

# 客户端类
class Client:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model: Optional[ViTModel] = None
        self.name: Optional[str] = None
    
    def set_model(self, model: ViTModel, device: str, name: str=None):
        self.model = model
        self.model.to(device)
        self.model.setName(name)
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
    
    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str, num_epochs: int):
        self.model.to(device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        total_loss = 0.0
        for epoch in range(num_epochs):
            for images, labels in self.data_loader:
                optimizer.zero_grad()  # 每个批次前清零梯度
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()  # 计算当前批次的梯度
                thisLoss = loss.item()
                total_loss += thisLoss
                optimizer.step()
        
        # 计算梯度变化
        final_state_dict = self.model.state_dict()
        gradient_changes = {}
        for key in self.initial_state_dict:
            gradient_changes[key] = final_state_dict[key] - self.initial_state_dict[key]
        return gradient_changes, total_loss / (len(self.data_loader) * num_epochs)

    def getName(self) -> str:
        return self.model.getName()
    
"""
梯度上升攻击
TODO: 更智能的攻击（现在的梯度上升攻击太基础了）
"""
class Attack(Client):
    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)

    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str, num_epochs: int):
        gradient_changes, average_loss = super().compute_gradient(criterion, device, num_epochs)
        
        # 返回gradient_changes中所有值的`相反数*attack_rate`
        neg_gradient_changes = {key: -value * attack_rate for key, value in gradient_changes.items()}
        return neg_gradient_changes, average_loss

# 服务器类
class Server:
    def __init__(self, model: ViTModel, device: str):
        self.global_model = model
        self.global_model.to(device)
        self.device = device
    
    def distribute_model(self, clients: List[Client]):
        for th, client in enumerate(clients):
            client.set_model(copy.deepcopy(self.global_model), device=self.device, name=f'Client{th + 1}')
    
    def aggregate_gradients(self, grads_dict:dict):
        avg_grads = {}
        for _, grads in grads_dict.items():
            for key in grads:
                if key not in avg_grads:
                    avg_grads[key] = grads[key].clone() / len(grads_dict)
                else:
                    avg_grads[key] += grads[key] / len(grads_dict)
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

#异常检测（梯度分析类）
class GradientAnalyzer:
    def __init__(self, n_components=2, use_gpu=True, device=device):
        self.n_components = n_components
        self.use_gpu = use_gpu
        self.device = device
        self.pca = PCA(n_components=self.n_components)
    
    def find_gradients(self, grads_dict:dict):
        for i, grads in grads_dict.items():
            print(f"Gradients from list {i}:")
            for key in grads:
                grad = grads[key]
                # Warnging: 下面一行print会输出很多内容
                print(f"Key: {key}, Gradient shape: {grad.shape}, Gradient dtype: {grad.dtype}, Gradient values: {grad}")
    
    # TODO: 更精确的检测模型
    def find_useful_gradients(self, grads_dict: dict) -> Tuple[List, List]:
        useful_grads_list = []
        anomalous_grads_list = []
        
        # Collect all gradients into a single numpy array
        grads_dict_cpu = {layer: {name: grad.cpu() for name, grad in grads.items()} for layer, grads in grads_dict.items()}
        all_grads = np.array([np.concatenate([grad.flatten() for grad in grads.values()]) for _, grads in grads_dict_cpu.items()])
        
        # Perform PCA on all gradients
        print(f"PCA Begin | {getNow()}")
        reduced_grads = self.pca.fit_transform(all_grads)
        print(f"PCA End | {getNow()}")
        
        # Calculate distances of each gradient to the principal components in PCA space
        distances = np.linalg.norm(reduced_grads - np.mean(reduced_grads, axis=0), axis=1)

        # Define anomaly threshold, e.g., 3 times standard deviation
        threshold = np.mean(distances) + PCA_rate * np.std(distances)

        # Determine useful and anomalous gradients
        for i, distance in enumerate(distances):
            if distance > threshold:
                anomalous_grads_list.append(i)  # 标记为异常的grad
            else:
                useful_grads_list.append(i)     # 标记为有用的grad
        print(anomalous_grads_list)
        return useful_grads_list, anomalous_grads_list
    
    #清除anomalous_grads_list中的数据
    def clean_grads(self, grads_dict: dict, anomalous_grads_list: list) -> dict:
        cleaned_grads_dict = {}
        current_index = 0
        for name, grads in grads_dict.items():
            if current_index not in anomalous_grads_list:
                cleaned_grads_dict[name] = grads
            else: 
                print(f'{name} is BANNED!')
            current_index += 1
        return cleaned_grads_dict
            
# 初始化数据管理器
data_manager = DataManager(num_clients, batch_size, datasize_perclient, datasize_valide)

# 初始化服务器
global_model = ViTModel(device=device, name='GlobalModel')
server = Server(global_model, device)

# 联邦学习过程
criterion = nn.CrossEntropyLoss()

ploter = Ploter(x='batch', y=['loss', 'accuracy'], title='loss and accuracy', filename=f'./result/{now}/lossAndAccuracy.svg')
accuracy = server.evaluate(data_manager.get_val_loader(), device)
timeRecorder.addRecord(f'init accuracy: {accuracy*100:.2f}%')
import math  # TODO: 计算真正的loss
ploter.addData(x=0, y={'loss': math.nan, 'accuracy': accuracy})

for round_num in range(num_rounds):
    timeRecorder.addRecord(f'Round {round_num + 1} of {num_rounds}')
    
    # 获取当前轮次的客户端数据加载器
    clients_data_loaders = data_manager.get_clients_data_loaders()
    clients = [Client(data_loader) for data_loader in clients_data_loaders]
    for attackerIndex in attackList:
        clients[attackerIndex] = Attack(clients_data_loaders[attackerIndex])

    # 分发当前的全局模型给所有客户端
    server.distribute_model(clients)
    # 每个客户端计算梯度
    grads_dict = {}
    total_loss = 0.0
    for th, client in enumerate(clients):
        # timeRecorder.addRecord(f'Round {round_num + 1}/{num_rounds} client {th + 1}/{num_clients} is computing gradients...')
        grads, loss = client.compute_gradient(criterion, device, epoch_client)
        grads_dict[client.getName()] = grads 
        total_loss += loss
    avg_loss = total_loss / num_clients
    print(f"Average loss: {avg_loss} | {getNow()}")
    
    # 服务器聚合梯度并更新全局模型
    if attackList and ifPCA:
        gradentAnalyzer = GradientAnalyzer()
        # gradentAnalyzer.find_gradients(grads_dict)
        _, anomaList = gradentAnalyzer.find_useful_gradients(grads_dict)
    if ifCleanAnoma and ifPCA and attackList:
        grads_dict = gradentAnalyzer.clean_grads(grads_dict, anomaList)
    
    avg_grads = server.aggregate_gradients(grads_dict)
    server.update_model(avg_grads)
    
    # 每轮训练后从测试集中随机挑选数据进行验证
    val_loader = data_manager.get_val_loader()
    # print(f'Begin to evaluate accuracy... | {getNow()}')
    accuracy = server.evaluate(val_loader, device)
    timeRecorder.addRecord(f"Round {round_num + 1}\'s accuracy: {accuracy * 100:.2f}%")
    ploter.addData(x=round_num + 1, y={'loss': avg_loss, 'accuracy': accuracy})

print("Federated learning completed.")
timeRecorder.printAll()
