'''
Author: LetMeFly
Date: 2024-07-06 11:37:16
LastEditors: LetMeFly
LastEditTime: 2024-07-06 12:10:33
'''
class Config:
    def __init__(self):
        self.num_clients = 10          # 客户端数量
        self.batch_size = 32           # 每批次多少张图片
        self.num_rounds = 32           # 总轮次
        self.epoch_client = 1          # 每个客户端的轮次
        self.datasize_perclient = 32   # 每个客户端的数据量
        self.datasize_valide = 1000    # 测试集大小
        self.learning_rate = 0.001     # 步长
        self.ifPCA = False             # 是否启用PCA评价 
        self.ifCleanAnoma = True       # 是否清理PCA抓出的异常数据
        self.PCA_rate = 1              # PCA偏离倍数
        self.attackList = [0, 1, 2]    # 恶意客户端下标
        self.attack_rate = 1           # 攻击强度
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.saveConfig()
    
    def saveConfig(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

config = Config()



import argparse

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # 动态添加参数解析
    known_args, unknown_args = parser.parse_known_args()
    
    parameters = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key_value = arg.lstrip("--").split("=", 1)
            if len(key_value) == 2:
                value = key_value[1]
                try:
                    value = eval(value)
                except:
                    pass
                parameters[key_value[0]] = value
            else:
                print(f"Invalid parameter format: {arg}")
    
    for key, value in parameters.items():
        print(f"{key}: {value}")
        print(type(value))

if __name__ == "__main__":
    """
    python try-config.py --name=9 --time=sfjlsj  --li="[1, 3, 8]"  --li2="[1, [3, 5], 's']"
    """
    main()

