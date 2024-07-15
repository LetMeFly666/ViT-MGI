'''
Author: LetMeFly666 814114971@qq.com
Date: 2024-07-13 09:50:07
LastEditors: LetMeFly
LastEditTime: 2024-07-15 18:06:19
FilePath: /master/makeTable.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE'
'''
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

def create_results_table(data, save_path='results_table.pdf'):
    Defend_method = ['Both-layer', 'Both-only', 'Both-pooling', 'PCA-layer', 'PCA-only', 'PCA-pooling', 'Forest-layer', 'Forest-pooling']

    # 计算每组数据的指标
    results = []
    for group in data:
        TP, FP, FN, TN ,TimesConsume= group
        recall = round(TP / (TP + FN), 3) if (TP + FN) != 0 else 0
        precision = round(TP / (TP + FP), 3) if (TP + FP) != 0 else 0
        f1_score = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) != 0 else 0
        accuracy = round((TP + TN) / (TP + FP + TN + FN), 3) if (TP + FP + TN + FN) != 0 else 0
        results.append([recall, precision, f1_score, accuracy, TimesConsume])

    # 将 Defend_method 插入到 results 的第一列
    for i, method in enumerate(Defend_method):
        results[i].insert(0, method)

    # 设置表格内容
    columns = ['Defend Method', 'Recall', 'Precision', 'F1 Score', 'Accuracy', 'Time']

    # 创建表格
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # 调整表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(2.0, 2.0)

    # 调整第一列宽度
    for key, cell in table.get_celld().items():
        if key[1] == 0:
            cell.set_width(0.8)
        else:
           cell.set_width(0.5)
    # 保存表格到文件
    plt.savefig(save_path)
    

def main():
    Attack_method = ['grad', 'label', 'backdoor']
    Defend_method = ['Both-layer', 'Both-only', 'Both-pooling', 'PCA-layer', 'PCA-only', 'PCA-pooling', 'Forest-layer', 'Forest-pooling']

    FilePath = './result/FinalCompare'

    for attack in Attack_method:
        data = []
        for defend in Defend_method:
            folder_path = os.path.join(FilePath, attack, defend)
            subfolder_names = os.listdir(folder_path)
            folder_path = os.path.join(folder_path, subfolder_names[0])
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        content = file.read()
                        received = re.findall(r'Correctly received: (\d+)', content)
                        wrongly_received = re.findall(r'Wrongly received: (\d+)', content)
                        correctly_rejected = re.findall(r'Correctly rejected: (\d+)', content)
                        wrongly_rejected = re.findall(r'Wrongly rejected: (\d+)', content)
                        start_time_str = re.findall(r"\| 00: Start \| (.*?) \|", content)
                        end_time_str = re.findall(r"\| 33: Round 32's accuracy: .*? \| (.*?) \|", content)
                        if start_time_str and end_time_str:
                            start_time_0 = start_time_str[0].strip()
                            end_time_0 = end_time_str[0].strip()
                        # 将字符串转换为datetime对象
                        start_time = datetime.strptime(start_time_0, '%Y.%m.%d-%H:%M:%S')
                        end_time = datetime.strptime(end_time_0, '%Y.%m.%d-%H:%M:%S')
                        # 计算时间差并转换为秒
                        time_diff = (end_time - start_time).total_seconds()
                        if received and wrongly_received and correctly_rejected and wrongly_rejected:
                            Correctly_received = int(received[-1])
                            Wrongly_received = int(wrongly_received[-1])
                            Correctly_rejected = int(correctly_rejected[-1])
                            Wrongly_rejected = int(wrongly_rejected[-1])
                            data.append([Correctly_received, Wrongly_received, Correctly_rejected, Wrongly_rejected,time_diff])
                            break  # 假设文件夹中只有一个 txt 文件，找到并处理后跳出循环
        print(data)
        create_results_table(data, f'result/Archive002-somePic/PaperTables/{attack}WithTime.pdf')

if __name__ == '__main__':
    main()
