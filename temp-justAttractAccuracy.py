'''
Author: LetMeFly
Date: 2024-07-11 20:00:26
LastEditors: LetMeFly
LastEditTime: 2024-07-15 18:33:27
Description: 对比不防御状态下攻击力度与准确率结果的关系，以证明攻击有效
'''
"""
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[]" --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=1 --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=2 --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=3 --device="cuda:0"
python main.py --ifFindAttack=False --attackMethod=grad --attackList="[0, 1]" --attack_rate=4 --device="cuda:0"
"""
import re

log_0 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.11-20:21:13                                 |
| 01: init accuracy: 11.00% | 2024.07.11-20:21:21                 |
| 02: Round 1's accuracy: 19.00% | 2024.07.11-20:21:27            |
| 03: Round 2's accuracy: 31.40% | 2024.07.11-20:21:32            |
| 04: Round 3's accuracy: 44.00% | 2024.07.11-20:21:37            |
| 05: Round 4's accuracy: 57.50% | 2024.07.11-20:21:43            |
| 06: Round 5's accuracy: 67.60% | 2024.07.11-20:21:48            |
| 07: Round 6's accuracy: 71.90% | 2024.07.11-20:21:54            |
| 08: Round 7's accuracy: 75.70% | 2024.07.11-20:21:59            |
| 09: Round 8's accuracy: 84.20% | 2024.07.11-20:22:04            |
| 10: Round 9's accuracy: 83.00% | 2024.07.11-20:22:09            |
| 11: Round 10's accuracy: 89.80% | 2024.07.11-20:22:15           |
| 12: Round 11's accuracy: 88.00% | 2024.07.11-20:22:20           |
| 13: Round 12's accuracy: 90.30% | 2024.07.11-20:22:25           |
| 14: Round 13's accuracy: 91.10% | 2024.07.11-20:22:30           |
| 15: Round 14's accuracy: 90.60% | 2024.07.11-20:22:36           |
| 16: Round 15's accuracy: 91.30% | 2024.07.11-20:22:41           |
| 17: Round 16's accuracy: 93.90% | 2024.07.11-20:22:46           |
| 18: Round 17's accuracy: 93.80% | 2024.07.11-20:22:52           |
| 19: Round 18's accuracy: 93.90% | 2024.07.11-20:22:57           |
| 20: Round 19's accuracy: 95.60% | 2024.07.11-20:23:02           |
| 21: Round 20's accuracy: 94.60% | 2024.07.11-20:23:07           |
| 22: Round 21's accuracy: 96.40% | 2024.07.11-20:23:13           |
| 23: Round 22's accuracy: 95.60% | 2024.07.11-20:23:18           |
| 24: Round 23's accuracy: 94.70% | 2024.07.11-20:23:23           |
| 25: Round 24's accuracy: 95.50% | 2024.07.11-20:23:28           |
| 26: Round 25's accuracy: 95.90% | 2024.07.11-20:23:34           |
| 27: Round 26's accuracy: 95.60% | 2024.07.11-20:23:39           |
| 28: Round 27's accuracy: 95.20% | 2024.07.11-20:23:44           |
| 29: Round 28's accuracy: 95.10% | 2024.07.11-20:23:50           |
| 30: Round 29's accuracy: 96.00% | 2024.07.11-20:23:55           |
| 31: Round 30's accuracy: 95.70% | 2024.07.11-20:24:00           |
| 32: Round 31's accuracy: 96.60% | 2024.07.11-20:24:05           |
| 33: Round 32's accuracy: 96.90% | 2024.07.11-20:24:11           |
+-----------------------------------------------------------------+
"""

log_1 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.11-20:24:31                                 |
| 01: init accuracy: 10.90% | 2024.07.11-20:24:39                 |
| 02: Round 1's accuracy: 14.40% | 2024.07.11-20:24:44            |
| 03: Round 2's accuracy: 19.70% | 2024.07.11-20:24:49            |
| 04: Round 3's accuracy: 25.60% | 2024.07.11-20:24:55            |
| 05: Round 4's accuracy: 28.40% | 2024.07.11-20:25:00            |
| 06: Round 5's accuracy: 35.80% | 2024.07.11-20:25:05            |
| 07: Round 6's accuracy: 42.80% | 2024.07.11-20:25:10            |
| 08: Round 7's accuracy: 51.40% | 2024.07.11-20:25:16            |
| 09: Round 8's accuracy: 54.60% | 2024.07.11-20:25:21            |
| 10: Round 9's accuracy: 63.80% | 2024.07.11-20:25:26            |
| 11: Round 10's accuracy: 68.40% | 2024.07.11-20:25:32           |
| 12: Round 11's accuracy: 74.30% | 2024.07.11-20:25:37           |
| 13: Round 12's accuracy: 75.20% | 2024.07.11-20:25:42           |
| 14: Round 13's accuracy: 79.40% | 2024.07.11-20:25:47           |
| 15: Round 14's accuracy: 81.80% | 2024.07.11-20:25:53           |
| 16: Round 15's accuracy: 84.20% | 2024.07.11-20:25:58           |
| 17: Round 16's accuracy: 85.10% | 2024.07.11-20:26:03           |
| 18: Round 17's accuracy: 86.10% | 2024.07.11-20:26:08           |
| 19: Round 18's accuracy: 88.00% | 2024.07.11-20:26:13           |
| 20: Round 19's accuracy: 87.80% | 2024.07.11-20:26:19           |
| 21: Round 20's accuracy: 88.70% | 2024.07.11-20:26:24           |
| 22: Round 21's accuracy: 89.30% | 2024.07.11-20:26:29           |
| 23: Round 22's accuracy: 90.20% | 2024.07.11-20:26:35           |
| 24: Round 23's accuracy: 90.40% | 2024.07.11-20:26:40           |
| 25: Round 24's accuracy: 91.20% | 2024.07.11-20:26:45           |
| 26: Round 25's accuracy: 91.50% | 2024.07.11-20:26:50           |
| 27: Round 26's accuracy: 91.90% | 2024.07.11-20:26:56           |
| 28: Round 27's accuracy: 92.20% | 2024.07.11-20:27:01           |
| 29: Round 28's accuracy: 92.90% | 2024.07.11-20:27:07           |
| 30: Round 29's accuracy: 93.80% | 2024.07.11-20:27:12           |
| 31: Round 30's accuracy: 93.00% | 2024.07.11-20:27:17           |
| 32: Round 31's accuracy: 91.80% | 2024.07.11-20:27:23           |
| 33: Round 32's accuracy: 93.80% | 2024.07.11-20:27:28           |
+-----------------------------------------------------------------+
"""

log_2 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.11-20:28:33                                 |
| 01: init accuracy: 8.00% | 2024.07.11-20:28:41                  |
| 02: Round 1's accuracy: 11.60% | 2024.07.11-20:28:46            |
| 03: Round 2's accuracy: 16.20% | 2024.07.11-20:28:52            |
| 04: Round 3's accuracy: 17.10% | 2024.07.11-20:28:57            |
| 05: Round 4's accuracy: 19.30% | 2024.07.11-20:29:02            |
| 06: Round 5's accuracy: 21.30% | 2024.07.11-20:29:07            |
| 07: Round 6's accuracy: 25.60% | 2024.07.11-20:29:13            |
| 08: Round 7's accuracy: 31.10% | 2024.07.11-20:29:18            |
| 09: Round 8's accuracy: 31.30% | 2024.07.11-20:29:23            |
| 10: Round 9's accuracy: 35.90% | 2024.07.11-20:29:28            |
| 11: Round 10's accuracy: 42.50% | 2024.07.11-20:29:34           |
| 12: Round 11's accuracy: 44.00% | 2024.07.11-20:29:39           |
| 13: Round 12's accuracy: 50.50% | 2024.07.11-20:29:44           |
| 14: Round 13's accuracy: 54.90% | 2024.07.11-20:29:50           |
| 15: Round 14's accuracy: 60.30% | 2024.07.11-20:29:55           |
| 16: Round 15's accuracy: 58.10% | 2024.07.11-20:30:00           |
| 17: Round 16's accuracy: 65.30% | 2024.07.11-20:30:05           |
| 18: Round 17's accuracy: 63.90% | 2024.07.11-20:30:11           |
| 19: Round 18's accuracy: 65.90% | 2024.07.11-20:30:16           |
| 20: Round 19's accuracy: 69.40% | 2024.07.11-20:30:21           |
| 21: Round 20's accuracy: 69.90% | 2024.07.11-20:30:26           |
| 22: Round 21's accuracy: 74.90% | 2024.07.11-20:30:32           |
| 23: Round 22's accuracy: 77.60% | 2024.07.11-20:30:37           |
| 24: Round 23's accuracy: 81.20% | 2024.07.11-20:30:42           |
| 25: Round 24's accuracy: 79.90% | 2024.07.11-20:30:48           |
| 26: Round 25's accuracy: 82.60% | 2024.07.11-20:30:53           |
| 27: Round 26's accuracy: 81.40% | 2024.07.11-20:30:58           |
| 28: Round 27's accuracy: 82.10% | 2024.07.11-20:31:04           |
| 29: Round 28's accuracy: 82.90% | 2024.07.11-20:31:09           |
| 30: Round 29's accuracy: 82.70% | 2024.07.11-20:31:14           |
| 31: Round 30's accuracy: 83.20% | 2024.07.11-20:31:20           |
| 32: Round 31's accuracy: 82.00% | 2024.07.11-20:31:25           |
| 33: Round 32's accuracy: 84.50% | 2024.07.11-20:31:30           |
+-----------------------------------------------------------------+
"""

log_3 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.11-20:31:50                                 |
| 01: init accuracy: 8.40% | 2024.07.11-20:31:58                  |
| 02: Round 1's accuracy: 11.00% | 2024.07.11-20:32:03            |
| 03: Round 2's accuracy: 14.10% | 2024.07.11-20:32:08            |
| 04: Round 3's accuracy: 15.60% | 2024.07.11-20:32:14            |
| 05: Round 4's accuracy: 15.90% | 2024.07.11-20:32:19            |
| 06: Round 5's accuracy: 18.50% | 2024.07.11-20:32:24            |
| 07: Round 6's accuracy: 20.20% | 2024.07.11-20:32:30            |
| 08: Round 7's accuracy: 21.90% | 2024.07.11-20:32:35            |
| 09: Round 8's accuracy: 25.20% | 2024.07.11-20:32:40            |
| 10: Round 9's accuracy: 30.70% | 2024.07.11-20:32:45            |
| 11: Round 10's accuracy: 30.80% | 2024.07.11-20:32:51           |
| 12: Round 11's accuracy: 31.20% | 2024.07.11-20:32:56           |
| 13: Round 12's accuracy: 35.50% | 2024.07.11-20:33:01           |
| 14: Round 13's accuracy: 37.50% | 2024.07.11-20:33:06           |
| 15: Round 14's accuracy: 38.80% | 2024.07.11-20:33:12           |
| 16: Round 15's accuracy: 38.00% | 2024.07.11-20:33:17           |
| 17: Round 16's accuracy: 38.90% | 2024.07.11-20:33:22           |
| 18: Round 17's accuracy: 43.30% | 2024.07.11-20:33:28           |
| 19: Round 18's accuracy: 44.00% | 2024.07.11-20:33:33           |
| 20: Round 19's accuracy: 43.70% | 2024.07.11-20:33:38           |
| 21: Round 20's accuracy: 44.30% | 2024.07.11-20:33:44           |
| 22: Round 21's accuracy: 44.90% | 2024.07.11-20:33:49           |
| 23: Round 22's accuracy: 47.10% | 2024.07.11-20:33:54           |
| 24: Round 23's accuracy: 50.30% | 2024.07.11-20:34:00           |
| 25: Round 24's accuracy: 52.10% | 2024.07.11-20:34:05           |
| 26: Round 25's accuracy: 53.70% | 2024.07.11-20:34:10           |
| 27: Round 26's accuracy: 56.00% | 2024.07.11-20:34:15           |
| 28: Round 27's accuracy: 56.20% | 2024.07.11-20:34:21           |
| 29: Round 28's accuracy: 59.10% | 2024.07.11-20:34:26           |
| 30: Round 29's accuracy: 60.00% | 2024.07.11-20:34:31           |
| 31: Round 30's accuracy: 57.90% | 2024.07.11-20:34:37           |
| 32: Round 31's accuracy: 59.20% | 2024.07.11-20:34:42           |
| 33: Round 32's accuracy: 62.70% | 2024.07.11-20:34:47           |
+-----------------------------------------------------------------+
"""

log_4 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.11-20:35:44                                 |
| 01: init accuracy: 5.70% | 2024.07.11-20:35:51                  |
| 02: Round 1's accuracy: 4.50% | 2024.07.11-20:35:57             |
| 03: Round 2's accuracy: 5.50% | 2024.07.11-20:36:02             |
| 04: Round 3's accuracy: 5.00% | 2024.07.11-20:36:07             |
| 05: Round 4's accuracy: 5.40% | 2024.07.11-20:36:12             |
| 06: Round 5's accuracy: 6.00% | 2024.07.11-20:36:18             |
| 07: Round 6's accuracy: 5.40% | 2024.07.11-20:36:23             |
| 08: Round 7's accuracy: 5.90% | 2024.07.11-20:36:28             |
| 09: Round 8's accuracy: 5.60% | 2024.07.11-20:36:33             |
| 10: Round 9's accuracy: 6.50% | 2024.07.11-20:36:39             |
| 11: Round 10's accuracy: 6.10% | 2024.07.11-20:36:44            |
| 12: Round 11's accuracy: 7.40% | 2024.07.11-20:36:49            |
| 13: Round 12's accuracy: 6.20% | 2024.07.11-20:36:54            |
| 14: Round 13's accuracy: 6.60% | 2024.07.11-20:37:00            |
| 15: Round 14's accuracy: 7.20% | 2024.07.11-20:37:05            |
| 16: Round 15's accuracy: 6.10% | 2024.07.11-20:37:10            |
| 17: Round 16's accuracy: 6.50% | 2024.07.11-20:37:15            |
| 18: Round 17's accuracy: 6.90% | 2024.07.11-20:37:21            |
| 19: Round 18's accuracy: 6.30% | 2024.07.11-20:37:26            |
| 20: Round 19's accuracy: 5.70% | 2024.07.11-20:37:31            |
| 21: Round 20's accuracy: 7.50% | 2024.07.11-20:37:36            |
| 22: Round 21's accuracy: 8.30% | 2024.07.11-20:37:41            |
| 23: Round 22's accuracy: 6.90% | 2024.07.11-20:37:47            |
| 24: Round 23's accuracy: 7.30% | 2024.07.11-20:37:52            |
| 25: Round 24's accuracy: 7.20% | 2024.07.11-20:37:57            |
| 26: Round 25's accuracy: 7.30% | 2024.07.11-20:38:03            |
| 27: Round 26's accuracy: 7.20% | 2024.07.11-20:38:08            |
| 28: Round 27's accuracy: 8.50% | 2024.07.11-20:38:13            |
| 29: Round 28's accuracy: 5.80% | 2024.07.11-20:38:19            |
| 30: Round 29's accuracy: 9.10% | 2024.07.11-20:38:24            |
| 31: Round 30's accuracy: 8.40% | 2024.07.11-20:38:29            |
| 32: Round 31's accuracy: 10.70% | 2024.07.11-20:38:34           |
| 33: Round 32's accuracy: 9.00% | 2024.07.11-20:38:40            |
+-----------------------------------------------------------------+
"""

# 使用正则表达式提取32次准确率
accuracies0 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_0)]
accuracies1 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_1)]
accuracies2 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_2)]
accuracies3 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_3)]
accuracies4 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_4)]

assert(len(accuracies1) == len(accuracies2) == len(accuracies3) == 32)

# 转换为浮点数

print(accuracies0)
print(accuracies1)
print(accuracies2)
print(accuracies3)
print(accuracies4)


import matplotlib.pyplot as plt
import numpy as np


# 将所有实验数据存储在一个列表中
all_accuracies = [accuracies0, accuracies1, accuracies2, accuracies3, accuracies4]
for acclist in all_accuracies:
    for i, val in enumerate(acclist):
        acclist[i] = val * 0.01

# 实验标签
labels = ['No Attacker', 'Intensity=1', 'Intensity=2', 'Intensity=3', 'Intensity=4']

plt.rcParams.update({'font.size': 24})  # 增大字体大小
plt.figure(figsize=(12, 6))

# 绘制图形
for i, accuracies in enumerate(all_accuracies):
    plt.plot(range(1, 33), accuracies, marker='o', label=labels[i])

# 添加图例
plt.legend(fontsize=18, loc='upper left', framealpha=0.5)

# 添加轴标签和标题
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy for Gradient Ascent Attack', fontsize=20)

plt.tight_layout(pad=0)
# 显示网格
plt.grid(True, which='both', linestyle='--')
plt.xticks(range(1, 33, 2))  # 设置x轴刻度为每隔2个刻度
# 显示图形
plt.savefig('./result/Archive002-somePic/PaperExperiments/001-gradAttack-attackRate.pdf')

"""stdout:
[19.0, 31.4, 44.0, 57.5, 67.6, 71.9, 75.7, 84.2, 83.0, 89.8, 88.0, 90.3, 91.1, 90.6, 91.3, 93.9, 93.8, 93.9, 95.6, 94.6, 96.4, 95.6, 94.7, 95.5, 95.9, 95.6, 95.2, 95.1, 96.0, 95.7, 96.6, 96.9]
[14.4, 19.7, 25.6, 28.4, 35.8, 42.8, 51.4, 54.6, 63.8, 68.4, 74.3, 75.2, 79.4, 81.8, 84.2, 85.1, 86.1, 88.0, 87.8, 88.7, 89.3, 90.2, 90.4, 91.2, 91.5, 91.9, 92.2, 92.9, 93.8, 93.0, 91.8, 93.8]
[11.6, 16.2, 17.1, 19.3, 21.3, 25.6, 31.1, 31.3, 35.9, 42.5, 44.0, 50.5, 54.9, 60.3, 58.1, 65.3, 63.9, 65.9, 69.4, 69.9, 74.9, 77.6, 81.2, 79.9, 82.6, 81.4, 82.1, 82.9, 82.7, 83.2, 82.0, 84.5]
[11.0, 14.1, 15.6, 15.9, 18.5, 20.2, 21.9, 25.2, 30.7, 30.8, 31.2, 35.5, 37.5, 38.8, 38.0, 38.9, 43.3, 44.0, 43.7, 44.3, 44.9, 47.1, 50.3, 52.1, 53.7, 56.0, 56.2, 59.1, 60.0, 57.9, 59.2, 62.7]
[4.5, 5.5, 5.0, 5.4, 6.0, 5.4, 5.9, 5.6, 6.5, 6.1, 7.4, 6.2, 6.6, 7.2, 6.1, 6.5, 6.9, 6.3, 5.7, 7.5, 8.3, 6.9, 7.3, 7.2, 7.3, 7.2, 8.5, 5.8, 9.1, 8.4, 10.7, 9.0]
"""