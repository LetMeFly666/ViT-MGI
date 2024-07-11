'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-10 20:16:56
LastEditors: LetMeFly
LastEditTime: 2024-07-11 09:41:59
FilePath: /master/src/attack.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
梯度上升攻击
TODO: 更智能的攻击（现在的梯度上升攻击太基础了）
"""
from src import Client
from torch.utils.data import DataLoader
import torch.nn as nn
from src import Config
import random
import torch

class GradientAscentAttack(Client):
    def __init__(self, data_loader: DataLoader, config: Config):
        self.config = config
        super().__init__(data_loader)

    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str, num_epochs: int):
        gradient_changes, average_loss = super().compute_gradient(criterion, device, num_epochs)

        # 返回gradient_changes中所有值的`相反数*attack_rate`
        attack_rate = self.config.attack_rate
        neg_gradient_changes = {key: -value * attack_rate for key, value in gradient_changes.items()}
        return neg_gradient_changes, average_loss


class LabelFlippingAttack(Client):
    def __init__(self, data_loader: DataLoader, config: Config):
        self.config = config
        super().__init__(data_loader)
    
    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str, num_epochs: int):
        # 打乱标签
        original_labels = []
        for _, labels in self.data_loader:
            original_labels.extend(labels.cpu().numpy())
        shuffled_labels = original_labels[:]
        # random.shuffle(shuffled_labels)
        for i in range(len(shuffled_labels)):
           if shuffled_labels[i] == 0:
               shuffled_labels[i] = 1
        modified_data_loader = [(images, torch.tensor(shuffled_labels[i:i + len(labels)])) 
                                for i, (images, labels) in enumerate(self.data_loader)]
        original_data_loader = self.data_loader
        self.data_loader = modified_data_loader
        gradient_changes, average_loss = super().compute_gradient(criterion, device, num_epochs)
        # 恢复原来的 data_loader
        self.data_loader = original_data_loader

        return gradient_changes, average_loss
    
class BackDoorAttack(Client):
    def __init__(self, data_loader: DataLoader, config: Config, trigger_size: int = 3):
        self.config = config
        self.trigger_size = trigger_size  # 触发模式的大小
        super().__init__(data_loader)
        
    def add_trigger(self, images: torch.Tensor) -> torch.Tensor:
        # 在图像的左下角添加触发模式
        images = images.clone()  # 避免修改原始图像
        images[:, :, -self.trigger_size:, :self.trigger_size] = 1  # 设置触发模式为全2，可以根据需要调整（后trigger_size行，前trigger_size列）
        return images
    
    def compute_gradient(self, criterion: nn.CrossEntropyLoss, device: str, num_epochs: int):
        # 获取原始的标签列表
        original_labels = []
        for _, labels in self.data_loader:
            original_labels.extend(labels.cpu().numpy())
        
        # 为每个图像添加触发模式，并修改标签为0
        modified_data_loader = [(self.add_trigger(images), torch.zeros_like(labels)) 
                                for images, labels in self.data_loader]
        
        original_data_loader = self.data_loader
        self.data_loader = modified_data_loader
        gradient_changes, average_loss = super().compute_gradient(criterion, device, num_epochs)
        self.data_loader = original_data_loader

        return gradient_changes, average_loss
        
  

