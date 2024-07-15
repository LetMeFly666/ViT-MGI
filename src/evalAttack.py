'''
Author: LetMeFly
Date: 2024-07-12 10:59:23
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-13 11:28:31
'''
from src import Client
from src import DataManager
from src import ViTModel
from src import Config
import numpy as np
import torch


class EvalLabelFlippingAttack():
    pass


class EvalBackdoorAttack(Client):
    def __init__(self, serverModel: ViTModel, data_loader: DataManager, config: Config):
        self.config = config
        self.serverModel = serverModel
        super().__init__(data_loader)
    
    def eval(self):
        original_images = []
        original_labels = []
        for images, labels in self.data_loader.get_val_loader():
            original_images.extend(images.cpu().numpy())
            original_labels.extend(labels.cpu().numpy())
            break

        # 转换为Tensor
        original_images = torch.tensor(original_images)
        original_labels = torch.tensor(original_labels)

        original_images = original_images.clone()  # 避免修改原始图像
        original_images[:, :, -3:, :3] = 1  # 设置触发模式为全2，可以根据需要调整（后trigger_size行，前trigger_size列）


        # 使用模型进行预测
        self.serverModel.to(self.config.device)
        self.serverModel.eval()

        with torch.no_grad():
            modified_images = original_images.to(self.config.device)
            outputs = self.serverModel(modified_images)
            _, predicted_labels = torch.max(outputs, 1)
            predicted_labels = predicted_labels.cpu().numpy()

        # 打印预测结果
        print(f"Original labels for modified images: {original_labels}")
        print(f"Predicted labels for modified images: {predicted_labels}")
        
        # 将 original_labels 转换为 numpy.ndarray
        original_labels = original_labels.numpy()

        # 比较修改后的标签与原始标签
        backdoor_success_rate = np.mean((predicted_labels == 0) & (original_labels != 0))
        print(f"Backdoor success rate: {backdoor_success_rate * 100:.2f}%")
        original_labels = torch.tensor(original_labels)
        # 还可以计算修改后图像的准确率
        correct_predictions = np.sum(predicted_labels == original_labels.cpu().numpy())
        total_predictions = len(original_labels)
        accuracy = correct_predictions / total_predictions
        print(f"Accuracy on modified images: {accuracy * 100:.2f}%")