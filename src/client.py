'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-10 20:21:51
LastEditors: LetMeFly
LastEditTime: 2024-07-11 09:34:54
FilePath: /master/src/client.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import copy
from torch.utils.data import DataLoader
from typing import Optional
import torch.optim as optim
import torch.nn as nn
from src import ViTModel

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
