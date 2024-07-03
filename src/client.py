'''
Author: LetMeFly
Date: 2024-07-02 23:14:49
LastEditors: LetMeFly
LastEditTime: 2024-07-03 09:46:10
'''
# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from typing import List, Tuple
from transformers import ViTForImageClassification, ViTConfig
from src.utils.data import get_dataloaders

class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for val, param in zip(parameters, self.model.parameters()):
            param.data = torch.tensor(val).to(param.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(None), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = self.model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def test(self):
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images).logits
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return loss / len(self.val_loader), accuracy


def main():
    train_loader, val_loader, _ = get_dataloaders()
    config = ViTConfig.from_pretrained('data/vit_base_patch16_224/config.json', num_labels=10)
    model = ViTForImageClassification.from_pretrained(
        'data/vit_base_patch16_224/pytorch_model.bin', 
        config=config, 
        ignore_mismatched_sizes=True
    ).cuda()
    client = CifarClient(model, train_loader, val_loader)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
