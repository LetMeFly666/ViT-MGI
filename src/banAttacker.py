'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-11 19:45:31
LastEditors: LetMeFly
LastEditTime: 2024-07-11 23:47:30
FilePath: /master/src/banAttacker.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from src import Config
from typing import List, Tuple, Dict

class BanAttacker:
    def __init__(self, config: Config):
        self.config = config
        self.num = config.num_clients
        self.userList = [0.5] * self.num  # 初始化每个用户异常的概率都为百分之50
        self.banList = []
        
    def Subjective_Logic_Model(self, estimate: List[float]) -> List[float]:
        updated_userList = []
        for i in range(self.num):
            if i < len(estimate):
                anomaly_score = estimate[i]
                current_score = self.userList[i]     
                # 根据隔离森林的结果更新用户评分
                # 这里假设隔离森林的评分越高越安全，越低越危险，可以根据具体情况调整条件判断逻辑
                if anomaly_score >= 0:  # 举例：如果隔离森林评分大于等于0.5，则认为较安全，可以增加评分
                    updated_score = min(current_score + anomaly_score / 2, 1.0)  # 增加0.1，但不超过1.0
                else:  # 否则认为较危险，减少评分
                    updated_score = max(current_score + anomaly_score / 2, 0.0)  # 减少0.1，但不低于0.0
                
                updated_userList.append(updated_score)
            else:
                updated_userList.append(self.userList[i])  # 如果没有对应的隔离森林评分，保持原始评分
        
        self.userList = updated_userList
        return self.userList
    
    def ban(self, roundNum: int) -> List[str]:
        for i, userScore in enumerate(self.userList):
            client_name = f"Client {i + 1}"
            if userScore <= 0.2 and client_name not in self.banList:
                self.banList.append(client_name)
                print(f"{client_name} has been banned in round {roundNum}")
        
        attack_list_clients = [f"Client {i + 1}" for i in self.config.attackList]
        if set(self.banList) == set(attack_list_clients):
            print(f"All attackers have been banned in round {roundNum}")

        return self.banList