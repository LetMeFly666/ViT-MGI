from typing import List, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from src import Config
import torch
import gc
import numpy as np
import datetime
from collections import defaultdict
from src import BanAttacker

getNow = lambda: datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')

#异常检测（梯度分析类）
class GradientAnalyzer:
    def __init__(self, config: Config, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.ban_history = []  # 封禁历史
        self.config=config
    
    def find_gradients(self, grads_dict: dict):
        toWrite = ''
        for i, grads in grads_dict.items():
            for key in grads:
                  grad = grads[key]
                  toWrite += f"Key: {key}, Gradient shape: {grad.shape}, Gradient dtype: {grad.dtype}, Gradient values: {grad}\n"

                  # Warnging: 下面一行print会输出很多内容
        with open(f'./result/grident-dict.txt', 'w') as f:
            f.write(toWrite)
                
    
    #将字典构建成列表
    def make_gradients(self, grads_dict: dict) -> np.ndarray:
        # Collect all gradients into a single numpy array
        grads_dict_cpu = {layer: {name: grad.cpu() for name, grad in grads.items()} for layer, grads in grads_dict.items()}
        all_grads = np.array([np.concatenate([grad.flatten() for grad in grads.values()]) for _, grads in grads_dict_cpu.items()]) # shape: (10, 85806346)
        # 尝试使用mle失败了，因为mle要求数据量大于特征数
        if self.config.ifPooling:
           print(f"pooling Begin | {getNow()}")
           all_grads = self.pooling(self.config.poolsize, all_grads)
           print(f"pooling End | {getNow()}")
        return all_grads
    
    
    def writeAnomalyScore(self, scoreList: list):
        with open(f'./result/Archive003-someText/AnomalyScore-{self.config.device}.txt', 'a') as f:
            try:
                self.config.Analyzer_alreadyWriteScore
            except:
                self.config.Analyzer_alreadyWriteScore = True
                f.write('*' * 20 + '\n')
                f.write(f'{getNow()}\n')
                f.write(f'{self.config.attackMethod}, {self.config.defendMethod}, {self.config.attackList}\n')
            f.write(f'{scoreList}\n')
    
    # TODO: 更精确的检测模型
    # 纯PCA，PCAForest是不会调用此函数的
    def PCA_Method(self, all_grads: List):
        useful_grads_list = []
        anomalous_grads_list = []
        # Perform PCA on all gradients
        print(f"PCA Begin | {getNow()}")
        reduced_grads = self.pca.fit_transform(all_grads)  # 两次fit_transform会使用不同的主成分，而一次fit_transform后调用transform会使用相同的主成分
        print(f"PCA End | {getNow()}")
        
        # Calculate distances of each gradient to the principal components in PCA space
        # 或者说，这里甚至已经不属于PCA的范畴了
        distances = np.linalg.norm(reduced_grads - np.mean(reduced_grads, axis=0), axis=1)

        # Define anomaly threshold, e.g., 3 times standard deviation
        threshold = np.mean(distances) + self.config.PCA_rate * np.std(distances)
        print(f'score: {distances}')
        self.writeAnomalyScore(list(distances))

        # Determine useful and anomalous gradients
        for i, distance in enumerate(distances):
            if distance > threshold:
                anomalous_grads_list.append(i)  # 标记为异常的grad
            else:
                useful_grads_list.append(i)     # 标记为有用的grad
        print(anomalous_grads_list)
        self.ban_history.append(anomalous_grads_list)
        return useful_grads_list, anomalous_grads_list
    
    def isolation_Forest_Method(self, all_grads: List[np.ndarray]) -> Tuple[List[int], List[int], List[float], List[int]]:
        useful_grads_list = []
        anomalous_grads_list = []
        
        print(f"Forest Begin | {getNow()}")
        isolation_forest = IsolationForest(n_estimators=self.config.forest_nEstimators, max_samples=1.0, max_features=1.0, random_state=42, contamination='auto')
        # Initialize Isolation Forest model

        # Fit the model to the gradients data
        isolation_forest.fit(all_grads)

        # Predict outliers (anomalies)
        outlier_preds = isolation_forest.predict(all_grads)  # 返回每个数据点是否为异常点（数组），-1表示异常，1表示正常

        # Get anomaly scores for each sample
        anomaly_scores = isolation_forest.decision_function(all_grads)
        self.writeAnomalyScore(list(anomaly_scores))

        # Print scores from high to low
        sorted_indices = np.argsort(anomaly_scores)[::-1]
        sorted_scores = anomaly_scores[sorted_indices]
        
        scoreStr = 'Anomaly scores (from high to low):\n'
        
        for idx, score in zip(sorted_indices, sorted_scores):
            scoreStr += f'Index: {idx}, Score: {score:.4f}\n'
        scoreStr = scoreStr[:-1]
        if self.config.isprintScore:
            print(scoreStr)
        # Extract useful and anomalous gradients based on predictions
        for idx, pred in enumerate(outlier_preds):
            if pred == -1:  # -1 indicates an outlier
                anomalous_grads_list.append(idx)
            else:
                useful_grads_list.append(idx)
        
        # print("Anomalous gradients:", anomalous_grads_list)
        self.ban_history.append(anomalous_grads_list)
        print(f"Forest End | {getNow()}")
        return useful_grads_list, anomalous_grads_list, anomaly_scores, sorted_indices
    
    def PCA_isolation_Forest_Method(self, all_grads: List[np.ndarray]) -> Tuple[List[int], List[int], List[float], List[int]]:
        useful_grads_list = []
        anomalous_grads_list = []
        
        # Perform PCA on all gradients
        # print(f"PCA Begin | {getNow()}")
        reduced_grads = self.pca.fit_transform(all_grads)  # 两次fit_transform会使用不同的主成分，而一次fit_transform后调用transform会使用相同的主成分
        # print(f"PCA End | {getNow()}")
        useful_grads_list, anomalous_grads_list, anomalous_scores, anomalous_indicates = self.isolation_Forest_Method(reduced_grads)            
        return useful_grads_list, anomalous_grads_list, anomalous_scores, anomalous_indicates

    # 如果是单独的隔离森林的话，暂无anomalous评分
    def find_useful_gradients(self, grads_dict: dict) -> Tuple[List[int], List[int], List[float], List[int]]:
        useful_grads_list = []
        anomalous_grads_list = []
        anomalous_scores = []
        anomalous_indicates=[]
        
        all_grads = self.make_gradients(grads_dict)
        print(all_grads.shape)
        
        if self.config.defendMethod == 'PCA':
            useful_grads_list, anomalous_grads_list = self.PCA_Method(all_grads)
        elif self.config.defendMethod == 'Forest':
           useful_grads_list, anomalous_grads_list, anomalous_scores, anomalous_indicates =self.isolation_Forest_Method(all_grads)
        else:  # Both
            useful_grads_list, anomalous_grads_list, anomalous_scores, anomalous_indicates = self.PCA_isolation_Forest_Method(all_grads)
        return useful_grads_list, anomalous_grads_list, anomalous_scores, anomalous_indicates
    
    def pooling(self, poolsize: int, all_grads: np.ndarray) -> np.ndarray:
        pooled_grads = []
        for grad in all_grads:
            grad_gpu = torch.tensor(grad, device=self.config.device)  # 将单个梯度移动到 GPU
            if grad_gpu.shape[0] % poolsize != 0:
                # 如果输入大小不能被 poolsize 整除，则进行填充
                padding_size = poolsize - (grad_gpu.shape[0] % poolsize)
                grad_gpu = torch.cat([grad_gpu, torch.zeros(padding_size, device=self.config.device)])
            
            grad_matrix = grad_gpu.view(-1, poolsize)  # 将梯度变为矩阵形式，每行有 poolsize 个元素
            if self.config.pooltype == 'Max':
                pooled_grad, _ = torch.max(grad_matrix, dim=1)  # 最大池化
            elif self.config.pooltype == 'Mean':
                pooled_grad = torch.mean(grad_matrix, dim=1)    # 平均池化
            elif self.config.pooltype == 'Sum':
                pooled_grad = torch.sum(grad_matrix, dim=1)     # 求和池化
            pooled_grads.append(pooled_grad.cpu().numpy())  # 将结果移动回 CPU 并转换为 NumPy 数组
            
            # 释放 GPU 内存
            del grad_gpu, grad_matrix, pooled_grad
            torch.cuda.empty_cache()
            gc.collect()  # 手动调用垃圾收集器

        pooled_grads = np.array(pooled_grads)
        print(f"poolsize: {poolsize}, pooled_grads shape: {pooled_grads.shape}, all_grads shape: {all_grads.shape}")
        return pooled_grads  # 相当于CPU内存又复制了一份，不影响原来的grad
     
    #清除anomalous_grads_list中的数据
    def clean_grads(self, grads_dict: dict, anomalous_grads_list: list,banList: list,userList:List) -> dict:
        cleaned_grads_dict = {}
        ban_indices = [int(client[6:]) - 1 for client in banList]  # 提取出 banList 中的编号，并减去 1 得到索引
        current_index = 0
        for name, grads in grads_dict.items():
            # 只要被ban了，就不要了；否则偶尔被评为恶意但是只要主观逻辑模型的总评分不低于0.6就要
            if (userList[current_index] >= 0.51 or current_index not in anomalous_grads_list) and current_index not in ban_indices:
                cleaned_grads_dict[name] = grads
                if current_index in self.config.attackList:
                    self.config.wrong_receive+=1
                else:
                    self.config.correct_receive+=1   
            else :
                if current_index in self.config.attackList:
                    self.config.wrong_reject+=1
                else:
                    self.config.correct_reject+=1
            current_index += 1
        # print(f"Correctly received: {self.config.correct_receive}")
        # print(f"Wrongly received: {self.config.wrong_receive}")
        # print(f"Correctly rejected: {self.config.correct_reject}")
        # print(f"Wrongly rejected: {self.config.wrong_reject}")
        return cleaned_grads_dict
    
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
        toSay += f'| {len(answer)}/{self.config.num_clients} | {self.config.attack_rate} | {self.config.PCA_rate} | {self.config.PCA_nComponents} | {len(self.ban_history)}次中有：'
        tempList = []
        for key, value in result.items():
            tempList.append((key, value))
        # TODO: 改成正常的
        tempList.sort(key=lambda x: (x[0][1], -x[0][0]))
        with open('./result/Archive003-someText/defendResult.txt', 'a') as f:
            f.write(f'{self.config.attackMethod}, {self.config.defendMethod}, {self.config.attackList} | {tempList}\n')
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
        toSay += f' <br/>{tempList} |\n'
        print(toSay)
        return result
