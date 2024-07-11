'''
Author: LetMeFly vme50ty
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly
LastEditTime: 2024-07-11 23:59:33
'''
import datetime
getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
now = getNow()
# del datetime
from src.utils import initPrint, TimeRecorder, Ploter
from typing import List, Optional, Tuple, Dict

initPrint(now)
print(now)

timeRecorder = TimeRecorder()
timeRecorder.addRecord('Start', getNow())



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transformsConfig
import torchvision.datasets as datasets
from transformers import ViTForImageClassification, ViTConfig
from transformers import ViTModel as ViTModel_Original
import numpy as np
import random
import copy
from sklearn.decomposition import PCA
import argparse
from collections import defaultdict
import gc
from sklearn.ensemble import IsolationForest
from src import Config, DataManager, ViTModel, Client, GradientAscentAttack, LabelFlippingAttack, BackDoorAttack, GradientAnalyzer, Server, FindLayer, BanAttacker


config = Config(now)

banAttacker = BanAttacker(config)

# 初始化数据管理器
data_manager = DataManager(config.num_clients, config.batch_size, config.datasize_perclient, config.datasize_valide)

# 初始化服务器
global_model = ViTModel(config, name='GlobalModel')
server = Server(global_model, config.device)

# 联邦学习过程
criterion = nn.CrossEntropyLoss()

# 攻击检测
gradentAnalyzer = GradientAnalyzer(config,n_components=config.PCA_nComponents)

findLayer = FindLayer(config, gradentAnalyzer)

ploter = Ploter(x='batch', y=['loss', 'accuracy'], title='loss and accuracy', filename=f'./result/{now}/lossAndAccuracy.svg')
accuracy = server.evaluate(data_manager.get_val_loader(), config.device)
timeRecorder.addRecord(f'init accuracy: {accuracy * 100:.2f}%')
import math  # TODO: 计算真正的loss
ploter.addData(x=0, y={'loss': math.nan, 'accuracy': accuracy})


with open('result/UsefulLayer/merge-resut.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]  # 去除空行和首尾空格
usefulLayers = set(lines)

banList=[]

for round_num in range(config.num_rounds):
    # timeRecorder.addRecord(f'Round {round_num + 1} of {config.num_rounds}')
    
    # 获取当前轮次的客户端数据加载器
    clients_data_loaders = data_manager.get_clients_data_loaders()
    clients = [Client(data_loader) for data_loader in clients_data_loaders]
    for attackerIndex in config.attackList:
        if config.attackMethod == 'backdoor':
            clients[attackerIndex] = BackDoorAttack(clients_data_loaders[attackerIndex],config)
        elif config.attackMethod == 'grad':
            clients[attackerIndex] = GradientAscentAttack(clients_data_loaders[attackerIndex],config)
        else:  # lable
            clients[attackerIndex] = LabelFlippingAttack(clients_data_loaders[attackerIndex],config)
        # TODO: 更多的攻击类型尝试
    # 分发当前的全局模型给所有客户端
    server.distribute_model(clients)
    # 每个客户端计算梯度
    grads_dict = {}  # 所有的梯度(weight)
    compute_grads_dict={}  # 提取的特征层的梯度
    total_loss = 0.0
    for th, client in enumerate(clients):
        # timeRecorder.addRecord(f'Round {round_num + 1}/{num_rounds} client {th + 1}/{num_clients} is computing gradients...')
        grads, loss = client.compute_gradient(criterion, config.device, config.epoch_client)
        grads_dict[client.getName()] = grads
        compute_grads_dict[client.getName()] = {k: v for k, v in grads.items() if k in usefulLayers}
        total_loss += loss
    avg_loss = total_loss / config.num_clients
    print(f"Average loss: {avg_loss} | {getNow()}")
    
    # 服务器聚合梯度并更新全局模型
    if config.ifFindAttack:
        #gradentAnalyzer.find_gradients(grads_dict)
        _, anomaList, anomaly_scores , _= gradentAnalyzer.find_useful_gradients(compute_grads_dict)
    if config.ifFindAttack and config.ifCleanAnoma:
        grads_dict = gradentAnalyzer.clean_grads(grads_dict, anomaList, banList, banAttacker.userList)
    #是否封禁被找出的用户
    if config.isBanAttacker and config.ifFindAttack:
        userList = banAttacker.Subjective_Logic_Model(anomaly_scores)
        print(userList)
        banList = banAttacker.ban(round_num)
    
    if config.ifFindUsefulLayer:
        values_list = findLayer.make_gradients_list(grads_dict)
        findLayer.find_useful_layer(values_list)
    
    avg_grads = server.aggregate_gradients(grads_dict)
    server.update_model(avg_grads)
    
    # 每轮训练后从测试集中随机挑选数据进行验证
    val_loader = data_manager.get_val_loader()
    # print(f'Begin to evaluate accuracy... | {getNow()}')
    accuracy = server.evaluate(val_loader, config.device)
    timeRecorder.addRecord(f"Round {round_num + 1}\'s accuracy: {accuracy * 100:.2f}%")
    ploter.addData(x=round_num + 1, y={'loss': avg_loss, 'accuracy': accuracy})

print(banList)

if config.attackMethod == 'backdoor':
    # model_path = f"./testModel_final.pth"
    # torch.save(server.model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    # 加载模型状态字典（如果需要在不同脚本或环境中加载）
    # server.model.load_state_dict(torch.load(model_path))
    # server.model.eval()

    # 获取验证集数据
    original_images = []
    original_labels = []
    for images, labels in data_manager.get_val_loader():
        original_images.extend(images.cpu().numpy())
        original_labels.extend(labels.cpu().numpy())
        break

    # 转换为Tensor
    original_images = torch.tensor(original_images)
    original_labels = torch.tensor(original_labels)

    original_images = original_images.clone()  # 避免修改原始图像
    original_images[:, :, -3:, :3] = 1  # 设置触发模式为全2，可以根据需要调整（后trigger_size行，前trigger_size列）


    # 使用模型进行预测
    card2 = config.device
    server.global_model.to(config.device)
    server.global_model.eval()

    with torch.no_grad():
        modified_images = original_images.to(config.device)
        outputs = server.global_model(modified_images)
        _, predicted_labels = torch.max(outputs, 1)
        predicted_labels = predicted_labels.cpu().numpy()

    # 打印预测结果
    print(f"Original labels for modified images: {original_labels}")
    print(f"Predicted labels for modified images: {predicted_labels}")

    # 比较修改后的标签与原始标签
    backdoor_success_rate = np.mean(predicted_labels == 0)  # 假设后门攻击将标签改为0
    print(f"Backdoor success rate: {backdoor_success_rate * 100:.2f}%")

    # 还可以计算修改后图像的准确率
    correct_predictions = np.sum(predicted_labels == original_labels.cpu().numpy())
    total_predictions = len(original_labels)
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on modified images: {accuracy * 100:.2f}%")


print("Federated learning completed.")
timeRecorder.printAll()
if config.ifFindAttack:
    gradentAnalyzer.evalBanAcc(config.attackList)
