'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-12 15:07:44
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-13 16:53:37
FilePath: /master/temp-justLabelsAttack.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import re
import matplotlib.pyplot as plt
import numpy as np


# 读取文件内容
with open('./result/2024.07.12-16:55:18/stdout.txt', 'r') as f:
    log_data = f.read()

round_accuracy = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_data)]
# 提取Misclassification ratio to 1
misclassification_ratio_to_1 = [float(ratio) for ratio in re.findall(r'Misclassification ratio to 1: (\d+\.\d+)', log_data)]

accuracies1 = round_accuracy[0:len(round_accuracy)//2]
attackSuccess = misclassification_ratio_to_1

accuracies1 = [val/100 for val in accuracies1]

with open('result/2024.07.13-16:33:24/stdout.txt', 'r') as f2:
    log_data_defend = f2.read()

defendAccuracies = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_data_defend)]
defend_misclassification_ratio_to_1 = [float(ratio) for ratio in re.findall(r'Misclassification ratio to 1: (\d+\.\d+)', log_data_defend)]

defendAccuracies = defendAccuracies[0:len(defendAccuracies)//2]
defendAccuracies = [val/100 for val in defendAccuracies]
defendAttackSuccess = [val-0.1 for val in defend_misclassification_ratio_to_1]

all_accuracies = [accuracies1, attackSuccess, defendAccuracies, defendAttackSuccess]

# 保留两位小数并将负数设置为0
all_accuracies = [[max(0, round(val, 2)) for val in lst] for lst in all_accuracies]

print(all_accuracies)

labels = ['Round Acc.', 'Label Success Rate', 'Round Acc. (def.)', 'Label Success Rate (def.)']

# 绘制图形
plt.rcParams.update({'font.size': 18})
# 创建图形
plt.figure(figsize=(12, 6))
markers = ['o', 'o', 'x', 'x']  # o: 圆圈, x: 叉号, s: 方块

for i, accuracies in enumerate(all_accuracies):
    plt.plot(range(1, 33), accuracies, marker=markers[i], label=labels[i])

# 添加图例
plt.legend(title='Experiments', loc='upper left')
plt.legend(fontsize=10)
# 添加轴标签和标题
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy over Training Rounds for LabelFlipping Attack Experiments')

plt.tight_layout(pad=0)
# 显示网格
plt.grid(True)
plt.xticks(range(1, 33))
# 显示图形
plt.savefig('./result/Archive002-somePic/PaperCompare/LabelFlippingAttack.pdf')
