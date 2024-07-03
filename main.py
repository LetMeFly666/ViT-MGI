'''
Author: LetMeFly
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-03 10:50:47
'''
import datetime
now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
del datetime
from src.utils import initPrint
from typing import List, Optional


import numpy as np

class FederatedModel:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
    
    def predict(self, x):
        return np.dot(x, self.weights)
    
    def update_weights(self, grad, learning_rate):
        self.weights -= learning_rate * grad

class Client:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.model: Optional[FederatedModel] = None
    
    def set_model(self, model):
        self.model = model
    
    def compute_gradient(self):
        predictions = self.model.predict(self.data)
        grad = np.dot(self.data.T, (predictions - self.labels)) / len(self.labels)
        return grad

class Server:
    def __init__(self, input_size, output_size, learning_rate):
        self.global_model = FederatedModel(input_size, output_size)
        self.learning_rate = learning_rate
    
    def distribute_model(self, clients: List[Client]):
        for client in clients:
            client.set_model(self.global_model)
    
    def aggregate_gradients(self, grads):
        return np.mean(grads, axis=0)
    
    def update_model(self, grads):
        aggregated_grad = self.aggregate_gradients(grads)
        self.global_model.update_weights(aggregated_grad, self.learning_rate)

def generate_data(num_samples, input_size, output_size):
    data = np.random.randn(num_samples, input_size)
    labels = np.dot(data, np.random.randn(input_size, output_size))
    return data, labels

# Parameters
num_clients = 5
input_size = 10
output_size = 1
learning_rate = 0.01
num_rounds = 10

if __name__ == "__main__":
    initPrint(now)
    print(now)

# Generate data for clients
clients: List[Client] = []
for _ in range(num_clients):
    data, labels = generate_data(100, input_size, output_size)
    clients.append(Client(data, labels))

# Initialize server
server = Server(input_size, output_size, learning_rate)

# Federated learning process
for round_num in range(num_rounds):
    print(f"Round {round_num+1}")
    
    # Distribute the current global model to all clients
    server.distribute_model(clients)
    
    # Each client computes the gradient
    gradients = []
    for th, client in enumerate(clients):
        print(f'Client{th} computing gradient...')
        grad = client.compute_gradient()
        gradients.append(grad)
    
    # Server aggregates the gradients and updates the global model
    server.update_model(gradients)

print("Federated learning completed.")

