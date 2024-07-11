'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-10 20:25:28
LastEditors: LetMeFly
LastEditTime: 2024-07-11 09:44:11
FilePath: /master/src/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from transformers import ViTForImageClassification, ViTConfig
from transformers import ViTModel as ViTModel_Original
import torch.nn as nn
from typing import List
from src import Config

# 定义ViT模型
class ViTModel(nn.Module):
    def __init__(self, config: Config, num_classes=10, name: str=None):
        super(ViTModel, self).__init__()
        self.config=config
        
        if self.config.ifPretrained:
            model_path = './data/vit_base_patch16_224'
            vit_config = ViTConfig.from_pretrained(model_path)
            self.model = ViTForImageClassification.from_pretrained(model_path, config=vit_config)
            self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        else:
            vit_config = ViTConfig()
            self.model = ViTModel_Original(vit_config)
            self.classifier = nn.Linear(vit_config.hidden_size, num_classes)
        self.model.to(self.config.device)  # 移动模型到设备
        self.name = 'defaultName'
    
    def forward(self, x):
        if self.config.ifPretrained:
            return self.model(x).logits
        else:
            outputs = self.model(x)
            logits = self.classifier(outputs.last_hidden_state[:, 0])
            return logits

    def setName(self, name: str) -> None:
        self.name = name
    
    def getName(self) -> str:
        return self.name
