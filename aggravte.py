'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-11 16:09:16
LastEditors: LetMeFly666 814114971@qq.com
LastEditTime: 2024-07-11 17:04:43
FilePath: /master/result/UsefulLayer/aggravte.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# 读取文件内容并转换为集合
def read_file_to_set(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]  # 去除空行和首尾空格
        return set(lines)

# 文件路径
file1 = 'result/UsefulLayer/test-backdoor.txt'
file2 = 'result/UsefulLayer/test-grad.txt'
file3 = 'result/UsefulLayer/test-label.txt'
file4 = 'result/UsefulLayer/merge-resut.txt'

# 读取文件内容到集合
set1 = read_file_to_set(file1)
set2 = read_file_to_set(file2)
set3 = read_file_to_set(file3)

# 找到交集
intersection_set = set1.intersection(set2, set3)

# 将交集结果存储到 file4
with open(file4, 'w') as f:
    for item in intersection_set:
        f.write(item + '\n')

# 输出交集结果
print("交集结果已存储到", file4)
