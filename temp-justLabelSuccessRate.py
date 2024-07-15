'''
Author: LetMeFly
Date: 2024-07-12 09:34:21
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-13 16:09:56
'''
import matplotlib.pyplot as plt

# 数据
labels = ['0', '1']
values = [8, 92]  # 假设100个本应为0的数据，其中92个被错误分类为1，8个被正确分类为0

# 绘制柱状图
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['green', 'red'])

# 添加数据标签
for i, value in enumerate(values):
    plt.text(i, value + 1, f'{value}%', ha='center', va='bottom')

# 添加标题和标签
plt.title('title')
plt.xlabel('result')
plt.ylabel('percent')

plt.xticks(range(1,33))
# 显示图形
plt.savefig('temp.png')
