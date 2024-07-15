'''
Author: LetMeFly
Date: 2024-07-12 11:18:42
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-13 16:12:41
'''
"""
python main.py --attackMethod=backdoor --ifFindAttack=False --attackList="[0, 1]"
"""
import re

with open('./result/2024.07.13-11:29:18/stdout.txt', 'r') as f:
    log_data = f.read()

# 提取Backdoor success rate
backdoor_success_rate = [float(acc) for acc in re.findall(r'Backdoor success rate: (\d+\.\d+)%', log_data)]

# 提取Accuracy on modified images
accuracy_on_modified_images = [float(acc) for acc in re.findall(r'Accuracy on modified images: (\d+\.\d+)%', log_data)]

# 提取Round *'s accuracy
round_accuracy = [float(acc) for acc in re.findall(r'Round \d+\'s accuracy: (\d+\.\d+)%', log_data)]
round_accuracy=round_accuracy[0:len(round_accuracy)//2]

with open('./result/2024.07.13-11:30:34/stdout.txt', 'r') as f:
    defend_log_data = f.read()

# 提取Backdoor success rate
defend_backdoor_success_rate = [float(acc) for acc in re.findall(r'Backdoor success rate: (\d+\.\d+)%', defend_log_data)]

# 提取Accuracy on modified images
defend_accuracy_on_modified_images = [float(acc) for acc in re.findall(r'Accuracy on modified images: (\d+\.\d+)%', defend_log_data)]

# 提取Round *'s accuracy
defend_round_accuracy = [float(acc) for acc in re.findall(r'Round \d+\'s accuracy: (\d+\.\d+)%', defend_log_data)]
defend_round_accuracy=defend_round_accuracy[0:len(defend_round_accuracy)//2]

# 打印结果
print("Backdoor success rate:", backdoor_success_rate)
print("Accuracy on modified images:", accuracy_on_modified_images)
print("Round accuracy:", round_accuracy)
print("defend Backdoor success rate:", defend_backdoor_success_rate)
print("defend Accuracy on modified images:", defend_accuracy_on_modified_images)
print("defend Round accuracy:", defend_round_accuracy)

"""stdout:
Backdoor success rate: [12.5, 25.0, 18.75, 46.88, 59.38, 65.62, 56.25, 71.88, 53.12, 65.62, 65.62, 68.75, 56.25, 46.88, 53.12, 56.25, 71.88, 62.5, 84.38, 71.88, 59.38, 65.62, 81.25, 75.0, 78.12, 84.38, 81.25, 93.75, 87.5, 87.5, 96.88, 84.38]
Accuracy on modified images: [6.25, 34.38, 25.0, 34.38, 28.12, 31.25, 40.62, 34.38, 40.62, 37.5, 37.5, 43.75, 53.12, 53.12, 50.0, 50.0, 37.5, 46.88, 21.88, 37.5, 46.88, 40.62, 28.12, 31.25, 34.38, 18.75, 25.0, 12.5, 12.5, 18.75, 18.75, 18.75]
Round accuracy: [13.5, 20.8, 29.0, 40.3, 47.3, 52.4, 54.9, 60.3, 65.1, 69.1, 73.6, 77.0, 77.7, 80.2, 83.0, 81.5, 83.7, 84.2, 86.1, 89.3, 88.1, 89.8, 89.2, 89.6, 89.9, 92.8, 91.4, 93.3, 92.8, 94.0, 94.2, 92.6]
"""


import matplotlib.pyplot as plt

# # 提取的数据
# backdoor_success_rate = [
#     12.50, 25.00, 18.75, 46.88, 59.38, 65.62, 56.25, 71.88,
#     53.12, 65.62, 65.62, 68.75, 56.25, 46.88, 53.12, 56.25,
#     71.88, 62.50, 84.38, 71.88, 59.38, 65.62, 81.25, 75.00,
#     78.12, 84.38, 81.25, 93.75, 87.50, 87.50, 96.88, 84.38
# ]

# accuracy_on_modified_images = [
#     6.25, 34.38, 25.00, 34.38, 28.12, 31.25, 40.62, 34.38,
#     40.62, 37.50, 37.50, 43.75, 53.12, 53.12, 50.00, 50.00,
#     37.50, 46.88, 21.88, 37.50, 46.88, 40.62, 28.12, 31.25,
#     34.38, 18.75, 25.00, 12.50, 12.50, 18.75, 18.75, 18.75
# ]

# round_accuracy = [
#     13.50, 20.80, 29.00, 40.30, 47.30, 52.40, 54.90, 60.30,
#     65.10, 69.10, 73.60, 77.00, 77.70, 80.20, 83.00, 81.50,
#     83.70, 84.20, 86.10, 89.30, 88.10, 89.80, 89.20, 89.60,
#     89.90, 92.80, 91.40, 93.30, 92.80, 94.00, 94.20, 92.60
# ]

rounds = list(range(1, 33))

# 图片字体大小
plt.rcParams.update({'font.size': 18})
# 创建图形
plt.figure(figsize=(12, 6))

# 绘制Round Accuracy
plt.plot(rounds, round_accuracy, label='Round Acc.', marker='o',markersize=3)

# plt.plot(rounds, defend_round_accuracy, label='Round Acc. (def.)', marker='o',markersize=3)

# 绘制Backdoor Success Rate
plt.plot(rounds, backdoor_success_rate, label='BD Success Rate', marker='x',markersize=3)

# plt.plot(rounds, defend_backdoor_success_rate, label='BD Success Rate (def.)', marker='x',markersize=3)

# 绘制Accuracy on Modified Images
plt.plot(rounds, accuracy_on_modified_images, label='Acc. on Mod. Images', marker='s',markersize=3)
# plt.plot(rounds, defend_accuracy_on_modified_images, label='Acc. on Mod. Images (def.)', marker='s',markersize=3)

# 添加图例
plt.legend(fontsize=10)

# 添加标题和标签
plt.title('Comparison of Metrics Over Rounds')
plt.xlabel('Rounds')
plt.ylabel('Percentage')

plt.tight_layout(pad=0)
"""
left: 子图左边的边距，作为图形宽度的比例（0 到 1 之间）。
right: 子图右边的边距，作为图形宽度的比例（0 到 1 之间）。
top: 子图顶部的边距，作为图形高度的比例（0 到 1 之间）。
bottom: 子图底部的边距，作为图形高度的比例（0 到 1 之间）
"""
# plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

# 显示图形
plt.grid(True)
plt.xticks(rounds)  # 设置x轴刻度为整数
plt.savefig('./result/Archive002-somePic/PaperExperiments/backdoorAttack.pdf')
