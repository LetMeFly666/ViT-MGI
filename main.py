'''
Author: LetMeFly vme50ty
Date: 2024-07-03 10:37:25
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-13 18:08:48
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
from src import Config, DataManager, ViTModel, Client, GradientAscentAttack, LabelFlippingAttack, BackDoorAttack, GradientAnalyzer, Server, FindLayer, BanAttacker, EvalBackdoorAttack


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
            clients[attackerIndex] = BackDoorAttack(clients_data_loaders[attackerIndex], config)
        elif config.attackMethod == 'grad':
            clients[attackerIndex] = GradientAscentAttack(clients_data_loaders[attackerIndex], config)
        else:  # lable
            clients[attackerIndex] = LabelFlippingAttack(clients_data_loaders[attackerIndex], config)
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
        if config.ifUsefulLayer:
             compute_grads_dict[client.getName()] = {k: v for k, v in grads.items() if k in usefulLayers}
        else:
            compute_grads_dict=grads_dict
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
    if config.isBanAttacker and config.ifFindAttack and config.defendMethod =='Both' :
        userList = banAttacker.Subjective_Logic_Model(anomaly_scores)
        # print(userList)
        # banList = banAttacker.ban(round_num)
    
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
    
print(f"Correctly received: {config.correct_receive}")
print(f"Wrongly received: {config.wrong_receive}")
print(f"Correctly rejected: {config.correct_reject}")
print(f"Wrongly rejected: {config.wrong_reject}")
    # # 如果攻击方法是 label
    # if config.attackMethod == 'lable':  # TODO: 与上面 backdoor 的代码重叠度较高
    #     correct_to_wrong = 0
    #     total_samples = 0
    #     all_labels = []
    #     all_predicted = []

    #     server.global_model.eval()
        
    #     for images, labels in val_loader:
    #         with torch.no_grad():
    #             images, labels = images.to(config.device), labels.to(config.device)
    #             outputs = server.global_model(images)
    #             _, predicted = torch.max(outputs, 1)
    #             all_labels.extend(labels.cpu().numpy())
    #             all_predicted.extend(predicted.cpu().numpy())
                
    #             # 统计被错误分类为1的样本数
    #             correct_to_wrong += (predicted == 1).sum().item()
                
    #             # 统计所有样本数
    #             total_samples += len(labels)
                
    #     print(f'Total samples: {total_samples}, Misclassified as 1: {correct_to_wrong}')
    #     # print('All labels:', all_labels)
    #     # print('All predicted:', all_predicted)
        
    #     if total_samples > 0:
    #         ratio = correct_to_wrong / total_samples
    #         print('Misclassification ratio to 1:', ratio)
    #     else:
    #         print('Error! No samples in the validation set.')
    
    
    #     if config.attackMethod == 'backdoor':
    #         backdoorAttackEvaler = EvalBackdoorAttack(server.global_model, data_manager, config)
    #         backdoorAttackEvaler.eval()

print(f'banList: {banList}')

    

print("Federated learning completed.")
timeRecorder.printAll()
if config.ifFindAttack:
    gradentAnalyzer.evalBanAcc(config.attackList)
