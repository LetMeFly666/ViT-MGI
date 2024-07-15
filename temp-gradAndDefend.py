'''
Author: LetMeFly
Date: 2024-07-11 20:00:26
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-13 16:05:24
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
| 00: Start | 2024.07.13-11:08:38                                 |
| 01: init accuracy: 18.30% | 2024.07.13-11:08:46                 |
| 02: Round 1's accuracy: 26.20% | 2024.07.13-11:08:51            |
| 03: Round 2's accuracy: 37.20% | 2024.07.13-11:08:56            |
| 04: Round 3's accuracy: 48.70% | 2024.07.13-11:09:02            |
| 05: Round 4's accuracy: 56.70% | 2024.07.13-11:09:07            |
| 06: Round 5's accuracy: 67.60% | 2024.07.13-11:09:12            |
| 07: Round 6's accuracy: 74.90% | 2024.07.13-11:09:17            |
| 08: Round 7's accuracy: 81.30% | 2024.07.13-11:09:23            |
| 09: Round 8's accuracy: 84.20% | 2024.07.13-11:09:28            |
| 10: Round 9's accuracy: 86.60% | 2024.07.13-11:09:33            |
| 11: Round 10's accuracy: 86.70% | 2024.07.13-11:09:38           |
| 12: Round 11's accuracy: 89.50% | 2024.07.13-11:09:43           |
| 13: Round 12's accuracy: 90.10% | 2024.07.13-11:09:49           |
| 14: Round 13's accuracy: 90.80% | 2024.07.13-11:09:54           |
| 15: Round 14's accuracy: 92.50% | 2024.07.13-11:09:59           |
| 16: Round 15's accuracy: 93.30% | 2024.07.13-11:10:04           |
| 17: Round 16's accuracy: 93.20% | 2024.07.13-11:10:10           |
| 18: Round 17's accuracy: 94.20% | 2024.07.13-11:10:15           |
| 19: Round 18's accuracy: 92.90% | 2024.07.13-11:10:20           |
| 20: Round 19's accuracy: 94.30% | 2024.07.13-11:10:25           |
| 21: Round 20's accuracy: 93.60% | 2024.07.13-11:10:31           |
| 22: Round 21's accuracy: 93.20% | 2024.07.13-11:10:36           |
| 23: Round 22's accuracy: 96.50% | 2024.07.13-11:10:41           |
| 24: Round 23's accuracy: 95.10% | 2024.07.13-11:10:46           |
| 25: Round 24's accuracy: 94.50% | 2024.07.13-11:10:51           |
| 26: Round 25's accuracy: 95.20% | 2024.07.13-11:10:57           |
| 27: Round 26's accuracy: 95.50% | 2024.07.13-11:11:02           |
| 28: Round 27's accuracy: 95.00% | 2024.07.13-11:11:07           |
| 29: Round 28's accuracy: 95.20% | 2024.07.13-11:11:12           |
| 30: Round 29's accuracy: 96.10% | 2024.07.13-11:11:18           |
| 31: Round 30's accuracy: 95.90% | 2024.07.13-11:11:23           |
| 32: Round 31's accuracy: 96.00% | 2024.07.13-11:11:28           |
| 33: Round 32's accuracy: 95.70% | 2024.07.13-11:11:33           |
+-----------------------------------------------------------------+
"""

log_1 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.13-11:02:36                                 |
| 01: init accuracy: 9.80% | 2024.07.13-11:02:44                  |
| 02: Round 1's accuracy: 19.60% | 2024.07.13-11:02:53            |
| 03: Round 2's accuracy: 29.60% | 2024.07.13-11:03:02            |
| 04: Round 3's accuracy: 41.30% | 2024.07.13-11:03:11            |
| 05: Round 4's accuracy: 49.90% | 2024.07.13-11:03:20            |
| 06: Round 5's accuracy: 60.90% | 2024.07.13-11:03:29            |
| 07: Round 6's accuracy: 69.90% | 2024.07.13-11:03:38            |
| 08: Round 7's accuracy: 74.10% | 2024.07.13-11:03:46            |
| 09: Round 8's accuracy: 79.00% | 2024.07.13-11:03:55            |
| 10: Round 9's accuracy: 81.80% | 2024.07.13-11:04:04            |
| 11: Round 10's accuracy: 86.20% | 2024.07.13-11:04:13           |
| 12: Round 11's accuracy: 86.80% | 2024.07.13-11:04:22           |
| 13: Round 12's accuracy: 88.60% | 2024.07.13-11:04:31           |
| 14: Round 13's accuracy: 89.60% | 2024.07.13-11:04:39           |
| 15: Round 14's accuracy: 91.10% | 2024.07.13-11:04:48           |
| 16: Round 15's accuracy: 91.70% | 2024.07.13-11:04:57           |
| 17: Round 16's accuracy: 93.00% | 2024.07.13-11:05:06           |
| 18: Round 17's accuracy: 90.70% | 2024.07.13-11:05:14           |
| 19: Round 18's accuracy: 93.20% | 2024.07.13-11:05:23           |
| 20: Round 19's accuracy: 94.20% | 2024.07.13-11:05:31           |
| 21: Round 20's accuracy: 93.10% | 2024.07.13-11:05:40           |
| 22: Round 21's accuracy: 94.20% | 2024.07.13-11:05:48           |
| 23: Round 22's accuracy: 95.90% | 2024.07.13-11:05:57           |
| 24: Round 23's accuracy: 94.60% | 2024.07.13-11:06:05           |
| 25: Round 24's accuracy: 93.90% | 2024.07.13-11:06:13           |
| 26: Round 25's accuracy: 94.70% | 2024.07.13-11:06:22           |
| 27: Round 26's accuracy: 94.50% | 2024.07.13-11:06:30           |
| 28: Round 27's accuracy: 95.80% | 2024.07.13-11:06:38           |
| 29: Round 28's accuracy: 95.00% | 2024.07.13-11:06:47           |
| 30: Round 29's accuracy: 95.80% | 2024.07.13-11:06:55           |
| 31: Round 30's accuracy: 95.50% | 2024.07.13-11:07:04           |
| 32: Round 31's accuracy: 95.80% | 2024.07.13-11:07:12           |
| 33: Round 32's accuracy: 95.70% | 2024.07.13-11:07:20           |
+-----------------------------------------------------------------+
"""

log_2 = """
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.13-11:01:56                                 |
| 01: init accuracy: 13.80% | 2024.07.13-11:02:04                 |
| 02: Round 1's accuracy: 14.90% | 2024.07.13-11:02:09            |
| 03: Round 2's accuracy: 20.90% | 2024.07.13-11:02:14            |
| 04: Round 3's accuracy: 27.40% | 2024.07.13-11:02:20            |
| 05: Round 4's accuracy: 35.50% | 2024.07.13-11:02:25            |
| 06: Round 5's accuracy: 43.10% | 2024.07.13-11:02:30            |
| 07: Round 6's accuracy: 50.40% | 2024.07.13-11:02:35            |
| 08: Round 7's accuracy: 56.50% | 2024.07.13-11:02:41            |
| 09: Round 8's accuracy: 62.80% | 2024.07.13-11:02:46            |
| 10: Round 9's accuracy: 67.30% | 2024.07.13-11:02:52            |
| 11: Round 10's accuracy: 72.60% | 2024.07.13-11:02:59           |
| 12: Round 11's accuracy: 74.20% | 2024.07.13-11:03:05           |
| 13: Round 12's accuracy: 74.70% | 2024.07.13-11:03:12           |
| 14: Round 13's accuracy: 79.80% | 2024.07.13-11:03:19           |
| 15: Round 14's accuracy: 80.40% | 2024.07.13-11:03:26           |
| 16: Round 15's accuracy: 83.40% | 2024.07.13-11:03:31           |
| 17: Round 16's accuracy: 85.70% | 2024.07.13-11:03:38           |
| 18: Round 17's accuracy: 84.90% | 2024.07.13-11:03:44           |
| 19: Round 18's accuracy: 85.90% | 2024.07.13-11:03:50           |
| 20: Round 19's accuracy: 87.90% | 2024.07.13-11:03:57           |
| 21: Round 20's accuracy: 89.50% | 2024.07.13-11:04:03           |
| 22: Round 21's accuracy: 90.20% | 2024.07.13-11:04:10           |
| 23: Round 22's accuracy: 90.40% | 2024.07.13-11:04:16           |
| 24: Round 23's accuracy: 91.60% | 2024.07.13-11:04:23           |
| 25: Round 24's accuracy: 91.40% | 2024.07.13-11:04:30           |
| 26: Round 25's accuracy: 92.30% | 2024.07.13-11:04:37           |
| 27: Round 26's accuracy: 92.30% | 2024.07.13-11:04:43           |
| 28: Round 27's accuracy: 94.00% | 2024.07.13-11:04:50           |
| 29: Round 28's accuracy: 93.70% | 2024.07.13-11:04:57           |
| 30: Round 29's accuracy: 93.50% | 2024.07.13-11:05:03           |
| 31: Round 30's accuracy: 92.20% | 2024.07.13-11:05:09           |
| 32: Round 31's accuracy: 93.20% | 2024.07.13-11:05:16           |
| 33: Round 32's accuracy: 92.00% | 2024.07.13-11:05:23           |
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
# accuracies3 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_3)]
# accuracies4 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_4)]

# assert(len(accuracies1) == len(accuracies2) == len(accuracies3) == 32)

# 转换为浮点数

print(accuracies0)
print(accuracies1)
print(accuracies2)
# print(accuracies3)
# print(accuracies4)


import matplotlib.pyplot as plt
import numpy as np


# 将所有实验数据存储在一个列表中
all_accuracies = [accuracies0, accuracies1, accuracies2]
for acclist in all_accuracies:
    for i, val in enumerate(acclist):
        acclist[i] = val * 0.01

# 实验标签
labels = ['No attacker', 'With defense', 'No defense']

rounds = list(range(1, 33))

# 绘制图形

# 图片字体大小
plt.rcParams.update({'font.size': 18})
# 创建图形
plt.figure(figsize=(12, 6))

for i, accuracies in enumerate(all_accuracies):
    plt.plot(range(1, 33), accuracies, marker='o', label=labels[i])

# 添加图例
plt.legend(title='Experiments')

plt.legend(fontsize=10)


# 添加轴标签和标题

plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy over Training Rounds for Gradient Ascent Attack Experiments')

plt.tight_layout(pad=0)

# 显示网格
plt.grid(True)

plt.xticks(rounds)  # 设置x轴刻度为整数
# 显示图形
plt.savefig('./result/Archive002-somePic/PaperCompare/gradAttack-attackRate=1.pdf')

"""stdout:
[19.0, 31.4, 44.0, 57.5, 67.6, 71.9, 75.7, 84.2, 83.0, 89.8, 88.0, 90.3, 91.1, 90.6, 91.3, 93.9, 93.8, 93.9, 95.6, 94.6, 96.4, 95.6, 94.7, 95.5, 95.9, 95.6, 95.2, 95.1, 96.0, 95.7, 96.6, 96.9]
[14.4, 19.7, 25.6, 28.4, 35.8, 42.8, 51.4, 54.6, 63.8, 68.4, 74.3, 75.2, 79.4, 81.8, 84.2, 85.1, 86.1, 88.0, 87.8, 88.7, 89.3, 90.2, 90.4, 91.2, 91.5, 91.9, 92.2, 92.9, 93.8, 93.0, 91.8, 93.8]
[11.6, 16.2, 17.1, 19.3, 21.3, 25.6, 31.1, 31.3, 35.9, 42.5, 44.0, 50.5, 54.9, 60.3, 58.1, 65.3, 63.9, 65.9, 69.4, 69.9, 74.9, 77.6, 81.2, 79.9, 82.6, 81.4, 82.1, 82.9, 82.7, 83.2, 82.0, 84.5]
[11.0, 14.1, 15.6, 15.9, 18.5, 20.2, 21.9, 25.2, 30.7, 30.8, 31.2, 35.5, 37.5, 38.8, 38.0, 38.9, 43.3, 44.0, 43.7, 44.3, 44.9, 47.1, 50.3, 52.1, 53.7, 56.0, 56.2, 59.1, 60.0, 57.9, 59.2, 62.7]
[4.5, 5.5, 5.0, 5.4, 6.0, 5.4, 5.9, 5.6, 6.5, 6.1, 7.4, 6.2, 6.6, 7.2, 6.1, 6.5, 6.9, 6.3, 5.7, 7.5, 8.3, 6.9, 7.3, 7.2, 7.3, 7.2, 8.5, 5.8, 9.1, 8.4, 10.7, 9.0]
"""