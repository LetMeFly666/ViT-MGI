'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-10 20:28:50
LastEditors: LetMeFly
LastEditTime: 2024-07-11 09:42:55
FilePath: /master/src/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import argparse

class Config:
    def __init__(self, now: str):
        self.now = now                 # 程序启动时间
        self.num_clients = 10          # 客户端数量
        self.batch_size = 32           # 每批次多少张图片
        self.num_rounds = 32           # 总轮次
        self.epoch_client = 1          # 每个客户端的轮次
        self.datasize_perclient = 32   # 每个客户端的数据量
        self.datasize_valide = 1000    # 测试集大小
        self.learning_rate = 0.001     # 步长
        self.ifFindAttack=True         # 是否启用找出攻击者
        self.ifCleanAnoma = True       # 是否清理PCA抓出的异常数据
        self.defendMethod = 'Both'     # 仅使用PCA评价，还是使用“PCA+隔离森林”：可选PCA或Both
        self.PCA_rate = 1              # PCA偏离倍数
        self.PCA_nComponents = 0.04    # PCA降维后的主成分数目
        self.forest_nEstimators = 300  # 随机森林的估计器数量
        self.attackList = [0, 1]       # 恶意客户端下标
        self.attack_rate = 1           # 梯度上升的攻击强度
        self.attackMethod = "backdoor" # 攻击方式：grad、lable、backdoor
        self.ifPooling = False         # 是否进行池化操作
        self.poolsize = 1000           # grads数组中每个grad，取n个数字中取最大值
        self.pooltype = "Max"          # 池化方式，可以为Mean或者Max，代表最大池化和平均池化
        self.ifPretrained = True       # 是否使用预训练模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parseAgrs()
        self.saveConfig()
    
    def parseAgrs(self):
        parser = argparse.ArgumentParser(description='FLDefinder Configuration')
        known_args, unknown_args = parser.parse_known_args()
        for arg in unknown_args:
            if arg.startswith("--"):
                key_value = arg.lstrip("--").split("=", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    try:
                        value = eval(value)
                    except:
                        pass
                    self.setConfig(key, value)
                else:
                    print(f"Invalid parameter format: {arg}")

    def saveConfig(self):
        toWrite = ''
        for key, value in self.__dict__.items():
            toWrite += f"{key} = {value}\n"
        with open(f'./result/{self.now}/config.env', 'w') as f:
            f.write(toWrite)
    
    def setConfig(self, key: str, value):
        self.__dict__[key] = value
