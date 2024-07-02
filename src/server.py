'''
Author: LetMeFly
Date: 2024-07-02 23:18:32
LastEditors: LetMeFly
LastEditTime: 2024-07-02 23:53:06
'''
# server.py
import torch
import torch.nn as nn
import flwr as fl
from src.utils.data import get_dataloaders
import timm
from typing import List, Tuple

def get_eval_fn(model):
    train_loader, val_loader, test_loader = get_dataloaders()
    
    def evaluate(weights: List) -> Tuple[float, dict]:
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return loss / len(test_loader), {"accuracy": accuracy}
    
    return evaluate

def main():
    print('begin to create model vit_base_patch16_224')
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10).cuda()
    print('create model vit_base_patch16_224 successfully')
    strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model))
    fl.server.start_server("localhost:8080", strategy=strategy, config={"num_rounds": 3})

if __name__ == "__main__":
    main()
