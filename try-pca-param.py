'''
Author: LetMeFly
Date: 2024-07-06 13:09:20
LastEditors: LetMeFly
LastEditTime: 2024-07-06 13:22:42
'''
# 示例数据
tempList = [
    ((1, 2), 3),
    ((3, 2), 3),
    ((2, 1), 4),
    ((2, 3), 1)
]

# 使用sort方法和lambda函数进行排序
tempList.sort(key=lambda x: (x[0][1], -x[0][0]))

# 输出排序结果
print(tempList)













from typing import List, Tuple, Dict
from collections import defaultdict

class Config:
    def __init__(self):
        self.num_clients = 10          # 客户端数量
        self.batch_size = 32           # 每批次多少张图片
        self.num_rounds = 32           # 总轮次
        self.epoch_client = 1          # 每个客户端的轮次
        self.datasize_perclient = 32   # 每个客户端的数据量
        self.datasize_valide = 1000    # 测试集大小
        self.learning_rate = 0.001     # 步长
        self.ifPCA = True              # 是否启用PCA评价 
        self.ifCleanAnoma = True       # 是否清理PCA抓出的异常数据
        self.PCA_rate = 1              # PCA偏离倍数
        self.PCA_nComponents = 2       # PCA降维后的主成分数目
        self.attackList = [0, 1, 2]    # 恶意客户端下标
        self.attack_rate = 1           # 攻击强度

config = Config()

#异常检测（梯度分析类）
class GradientAnalyzer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.ban_history = []  # 封禁历史
    
    def addBanList(self, banList: List[int]):
        self.ban_history.append(banList)
    
    def evalBanAcc(self, answer: List[int]) -> Dict[Tuple[int, int], int]:
        result = defaultdict(int)
        for banList in self.ban_history:
            correct, error = 0, 0  # 正确的，错抓的
            for thisClient in banList:
                if thisClient in answer:
                    correct += 1
                else:
                    error += 1
            thisState = (correct, error)
            result[thisState] += 1
        toSay = '| 攻击者 | 攻击力度 | PCA的偏离倍数 | PCA降维后的主成分数目 | 表现 |\n'
        toSay += '|---|---|---|---|---|\n'
        toSay += f'| {len(answer)}/{config.num_clients} | {config.attack_rate} | {config.PCA_rate} | {config.PCA_nComponents} | {len(self.ban_history)}次中有：'
        tempList = []
        for key, value in result.items():
            tempList.append((key, value))
        tempList.sort(key=lambda x: (x[0][1], -x[0][0]))
        for (correct, error), times in tempList:
            toSay += f'{times}次'
            if correct == len(answer) and not error:
                toSay += '完全正确'
            elif not error:
                toSay += f'少抓{len(answer) - correct}个'
            else:
                if correct < len(answer):
                    toSay += f'少抓{len(answer) - correct}个'
                toSay += f'多抓{error}个'
            toSay += '，'
        toSay = toSay[:-1]
        toSay += ' |\n'
        print(toSay)
        return result


# 初始化异常检测器
gradientAnalyzer = GradientAnalyzer()
"""
2次完全正确，3次少抓1个，4次少抓2个，5次少抓3个，1次多抓1个，1次少抓1个多抓1个，1次少抓3个多抓1个
"""
gradientAnalyzer.addBanList([0, 1, 2])
gradientAnalyzer.addBanList([0, 1])
gradientAnalyzer.addBanList([0, 1, 2])
gradientAnalyzer.addBanList([0, 1, 2, 3])
gradientAnalyzer.addBanList([1, 2])
gradientAnalyzer.addBanList([1, 2])
gradientAnalyzer.addBanList([1, 2, 3])
gradientAnalyzer.addBanList([1])
gradientAnalyzer.addBanList([1])
gradientAnalyzer.addBanList([1])
gradientAnalyzer.addBanList([1])
gradientAnalyzer.addBanList([])
gradientAnalyzer.addBanList([])
gradientAnalyzer.addBanList([5])
gradientAnalyzer.addBanList([])
gradientAnalyzer.addBanList([])
gradientAnalyzer.addBanList([])

gradientAnalyzer.evalBanAcc([0, 1, 2])
