'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-11 13:10:39
LastEditors: LetMeFly
LastEditTime: 2024-07-16 09:48:11
FilePath: /master/src/findLayer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from src import Config, GradientAnalyzer
from typing import List, Tuple, Dict
import numpy as np


#寻找各个Layer中，最有用的部分

class FindLayer:
    def __init__(self, config: Config, analyzer: GradientAnalyzer):
        self.usefulLayer = []   # 存储有用层的列表
        self.config = config
        self.analyzer = analyzer
    
    #将用户:gradents键值对转换为gradents列表。列表中的每一行由键值对组成，键值对为layer：值。
    def make_gradients_list(self, grads_dict: dict) -> List[Dict[str, List[float]]]:
        values_list = []
        # c1 = grads_dict['Client1']
        # print(len(c1))
        # layerList = ''
        # for layer_name, grad_tensor in c1.items():
        #     layerList += layer_name + '\n'
        # print(layerList)
        # exit(0)

        # Convert gradients to CPU and reshape into desired format
        for client_name, layer_grads in grads_dict.items():
            client_values = {}
            for layer_name, grad_tensor in layer_grads.items():
                flattened_grad = grad_tensor.cpu().flatten().tolist()
                client_values[layer_name] = flattened_grad
            values_list.append(client_values)

        return values_list

    def extract_layer_values(self, values_list: List[Dict[str, List[float]]], layer_name: str) -> np.ndarray:
        layer_values = []
        for values_dict in values_list:
            layer_values.append(values_dict[layer_name])
        layer_values_np = np.array(layer_values)
        print(f"Shape of layer_values_np for layer '{layer_name}': {layer_values_np.shape}")
        return layer_values_np
    
    def isGoodLayer(self, anomalous_indicates: List[int]) -> bool:
        # 获取anomalous_indicates中的后k位
        len_attack_list = len(self.config.attackList)
        last_k_indicates = set(anomalous_indicates[-len_attack_list:])
        attack_set = set(self.config.attackList)
       # 检查anomalous_indicates中的后k位是否与self.config.attackList中的值集合相同
        return last_k_indicates == attack_set
        
    def find_useful_layer(self, values_list: List[Dict[str, List[float]]]) -> List[str]:
        for layer_name in values_list[0].keys():
            tempVect = self.extract_layer_values(values_list, layer_name)
            _, _, anomalous_scores, anomalous_indicates = self.analyzer.PCA_isolation_Forest_Method(tempVect)
            if self.isGoodLayer(anomalous_indicates):
               self.usefulLayer.append(layer_name)
        self.writeLayerName(self.usefulLayer)
        return self.usefulLayer
    
    def writeLayerName(self, useful_layer_names: List[str]):
        with open(f'./result/Archive003-someText/UsefulLayer/test-{self.config.attackMethod}.txt', 'w') as f:
            for layer_name in useful_layer_names:
                f.write(layer_name + '\n')
