'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-10 21:22:06
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-10 21:23:36
FilePath: /master/src/sever.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from src import ViTModel
from typing import List
import copy
from src import Client
import torch

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
