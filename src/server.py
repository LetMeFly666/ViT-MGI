'''
Author: LetMeFly
Date: 2024-07-02 23:18:32
LastEditors: LetMeFly
LastEditTime: 2024-07-03 10:01:59
'''
# server.py
import torch
import torch.nn as nn
import flwr as fl
from src.utils.data import get_dataloaders
from transformers import ViTForImageClassification, ViTConfig
from typing import List, Tuple

def get_eval_fn(model):
    train_loader, val_loader, test_loader = get_dataloaders()
    
    def evaluate(server_round: int, parameters: List, config: dict) -> Tuple[float, dict]:
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images).logits
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return loss / len(test_loader), {"accuracy": accuracy}
    
    return evaluate

def main():
    # print('begin to create model vit_base_patch16_224')
    # model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10).cuda()
    # model.load_state_dict(torch.load('data/vit_base_patch16_224/pytorch_model.bin'))
    # print('create model vit_base_patch16_224 successfully')
    config = ViTConfig.from_pretrained('data/vit_base_patch16_224/config.json', num_labels=10)
    model = ViTForImageClassification.from_pretrained('data/vit_base_patch16_224/pytorch_model.bin', config=config, ignore_mismatched_sizes=True).cuda()
    strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model))
    server_config = fl.server.ServerConfig(num_rounds=3)
    fl.server.start_server(server_address="localhost:8080", strategy=strategy, config=server_config)

if __name__ == "__main__":
    main()
