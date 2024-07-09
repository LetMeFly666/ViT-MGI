import os
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

base_path = './result/Archive001-oldHistory/Archive008-nComponseAndForestNEstimators'
# base_path = './result'

def read_config(file_path: str) -> Dict[str, str]:
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config

def extract_detection_result(line: str) -> str:
    pattern = r'\|\s*\|\s*[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*[^|]*\|\s*([^|]+)\s*\|'
    match = re.search(pattern, line)
    if match:
        return match.group(1).strip()
    else:
        return ""

def extract_accuracies(log_file: str) -> Tuple[List[float], str, str, str]:
    accuracies = []
    detection_result = ""
    start_time = ""
    end_time = ""
    with open(log_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r"Round (\d+)'s accuracy: (\d+\.\d+)%", line)
            if match:
                round_num = int(match.group(1))
                accuracy = float(match.group(2))
                if (round_num, accuracy) not in accuracies:
                    accuracies.append((round_num, accuracy))
        for line in reversed(lines):
            if "次中有：" in line:
                detection_result = line.strip()
                detection_result = extract_detection_result(detection_result)
                break
        # 提取时间信息
        time_matches = re.findall(r'\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}', " ".join(lines))
        if time_matches:
            start_time = time_matches[0]
            end_time = time_matches[-1]
    accuracies = [accuracy for _, accuracy in sorted(accuracies)]
    return accuracies, detection_result, start_time, end_time

def get_max_accuracy(accuracies: List[float]) -> Tuple[float, int]:
    max_accuracy = max(accuracies)
    max_round = accuracies.index(max_accuracy) + 1
    return max_accuracy, max_round

def print_summary(config: Dict[str, str], accuracies: List[float], detection_result: str, start_time: str, end_time: str) -> str:
    pca_components = config.get('PCA_nComponents', 'N/A')
    forest_n_estimators = config.get('forest_nEstimators', 'N/A')
    accuracy_link = f"[准确率]({os.path.join(base_path, config['folder_name'] + '/accuracyList.txt')})"
    max_accuracy, max_round = get_max_accuracy(accuracies)
    duration = datetime.strptime(end_time, '%Y.%m.%d-%H:%M:%S') - datetime.strptime(start_time, '%Y.%m.%d-%H:%M:%S')
    result_img = f"![结果图]({os.path.join(base_path, config['folder_name'] + '/lossAndAccuracy.svg')})"
    
    detection_result_clean = detection_result.split(" <br/>")[0]

    return f"| {pca_components} | {forest_n_estimators} | {detection_result_clean} | {accuracy_link} | {max_accuracy}% | {max_round} | {duration} | {result_img} |"

def main():
    date_format = '%Y.%m.%d-%H:%M:%S'
    start_date = datetime.strptime('2024.07.09-00:28:55', date_format)
    end_date = datetime.strptime('2024.07.09-15:01:09', date_format)
    
    folder_names = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    folder_names = [f for f in folder_names if re.match(r'\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}$', f)]
    folder_names = [f for f in folder_names if start_date <= datetime.strptime(f[:19], date_format) <= end_date]

    table_header = "| PCA components | forest n estimators | 检测结果 | accuracy | 最大准确率 | 首次出现轮次 | 执行耗时 | 结果图 |\n"
    table_header += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    table_rows = []

    for folder_name in folder_names:
        config_path = os.path.join(base_path, folder_name, 'config.env')
        log_path = os.path.join(base_path, folder_name, 'stdout.txt')

        if os.path.exists(config_path) and os.path.exists(log_path):
            config = read_config(config_path)
            config['folder_name'] = folder_name

            accuracies, detection_result, start_time, end_time = extract_accuracies(log_path)

            with open(os.path.join(base_path, folder_name, 'accuracyList.txt'), 'w') as acc_file:
                acc_file.write("\n".join(map(str, accuracies)))

            row = print_summary(config, accuracies, detection_result, start_time, end_time)
            table_rows.append(row)
    
    table_rows = sorted(table_rows, key=lambda x: (
        float(re.search(r'\d*\.?\d+', x.split('|')[2]).group() if re.search(r'\d*\.?\d+', x.split('|')[2]) else 'inf'),  # forest n estimators小的优先
        -float(re.search(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?', x.split('|')[1]).group() if re.search(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?', x.split('|')[1]) else 'inf'),  # PCA components大的优先
        x.split('|')[0]  # 文件夹日期小的优先
    ))

    markdown_table = table_header + "\n".join(table_rows)
    print(markdown_table)

if __name__ == "__main__":
    main()
