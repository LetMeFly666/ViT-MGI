import os
import re
from datetime import datetime, timedelta

def read_config(folder_path):
    config_file_path = os.path.join(folder_path, 'config.env')
    config = {}
    with open(config_file_path, 'r') as file:
        for line in file:
            # 忽略空行和注释行
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config

def extract_accuracies(log_file_path):
    accuracies = []
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r"Round (\d+)'s accuracy: (\d+\.\d+)%", line)
            if match:
                round_num = int(match.group(1))
                accuracy = float(match.group(2))
                if (round_num, accuracy) not in accuracies:
                    accuracies.append((round_num, accuracy))
    return [accuracy for _, accuracy in sorted(accuracies)]

def get_max_accuracy(accuracies):
    max_accuracy = max(accuracies)
    max_round = accuracies.index(max_accuracy) + 1
    return max_accuracy, max_round

def extract_time_info(log_file_path):
    times = []
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'\| (\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}) \|', line)
            if match:
                time_str = match.group(1)
                time_obj = datetime.strptime(time_str, "%Y.%m.%d-%H:%M:%S")
                times.append(time_obj)
    if times:
        elapsed_time = max(times) - min(times)
    else:
        elapsed_time = timedelta(0)
    return elapsed_time

def generate_markdown_summary(folder_name, folder_path):
    config = read_config(folder_path)
    
    log_file_path = os.path.join(folder_path, 'stdout.txt')
    accuracies = extract_accuracies(log_file_path)
    elapsed_time = extract_time_info(log_file_path)
    
    max_accuracy, max_round = get_max_accuracy(accuracies)
    accuracy_str = ", ".join([f"{acc}%" for acc in accuracies])
    accuracy_html = f"<div style='overflow-x:auto;width:300px;'>{accuracy_str}</div>"
    
    keys_of_interest = ['epoch_client', 'learning_rate', 'batch_size', 'device']
    config_values = [config.get(key, 'N/A') for key in keys_of_interest]
    
    result_image = f"./result/{folder_name}/lossAndAccuracy.svg"
    result_image_markdown = f"![Result Image]({result_image})"
    
    values = {
        'epoch_client': int(config_values[0]),
        'learning_rate': float(config_values[1]),
        'batch_size': int(config_values[2]),
        'device': config_values[3],
        'accuracy_str': accuracy_html,
        'max_accuracy': max_accuracy,
        'max_round': max_round,
        'elapsed_time': str(elapsed_time),
        'result_image_markdown': result_image_markdown
    }
    
    return values

def is_within_date_range(folder_name, start_date, end_date):
    try:
        folder_date = datetime.strptime(folder_name, "%Y.%m.%d-%H:%M:%S")
        return start_date <= folder_date <= end_date
    except ValueError:
        return False

if __name__ == "__main__":
    result_folder_path = './result'  # 将此替换为实际的结果文件夹路径
    start_date_str = '2024.07.07-00:41:37'
    end_date_str = '2024.07.07-04:14:49'
    
    start_date = datetime.strptime(start_date_str, "%Y.%m.%d-%H:%M:%S")
    end_date = datetime.strptime(end_date_str, "%Y.%m.%d-%H:%M:%S")
    
    summaries = []
    for folder_name in os.listdir(result_folder_path):
        folder_path = os.path.join(result_folder_path, folder_name)
        if os.path.isdir(folder_path) and is_within_date_range(folder_name, start_date, end_date):
            summary = generate_markdown_summary(folder_name, folder_path)
            summaries.append(summary)
    
    # Sort the summaries based on the given criteria
    summaries.sort(key=lambda x: (x['device'], -x['epoch_client'], -x['learning_rate'], -x['batch_size']))
    
    header = "| 单个客户端训练轮次 | learning rate | batch size | device | accuracy | 最大准确率 | 最大准确率的首次出现轮次 | 程序执行耗时 | 结果图 |"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    print(header)
    print(separator)
    
    for summary in summaries:
        values = "| " + f"{summary['epoch_client']} | {summary['learning_rate']} | {summary['batch_size']} | {summary['device']} | {summary['accuracy_str']} | {summary['max_accuracy']}% | {summary['max_round']} | {summary['elapsed_time']} | {summary['result_image_markdown']} |"
        print(values)
    
    # Calculate the total elapsed time
    total_elapsed_time = end_date - start_date
    print(f"\n总程序执行耗时: {total_elapsed_time}")
