'''
Author: LetMeFly
Date: 2024-07-02 23:14:49
LastEditors: LetMeFly
LastEditTime: 2024-07-02 23:42:45
'''
# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from typing import List, Tuple
from timm import create_model
from src.utils.data import get_dataloaders

class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self) -> List:
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters: List) -> None:
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val).to(param.device)

    def fit(self, parameters: List, config: dict) -> Tuple[List, int, dict]:
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List, config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

    def train(self) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def test(self) -> Tuple[float, float]:
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return loss / len(self.val_loader), accuracy


def main():
    train_loader, val_loader, _ = get_dataloaders()
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=10).cuda()
    client = CifarClient(model, train_loader, val_loader)
    fl.client.start_numpy_client("localhost:8080", client)

if __name__ == "__main__":
    main()
