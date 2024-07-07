这是我的输出结果，请你写一个Python脚本，提炼出这个输出日志中关于准确率的有效信息

```
```








Traceback (most recent call last):
  File "extract-accuracy-inLog.py", line 26, in <module>
    accuracies = extract_accuracies(log_file)
  File "extract-accuracy-inLog.py", line 13, in extract_accuracies
    match = re.search(r'Round (\d+\'s accuracy: (\d+\.\d+)%', line)
  File "/home/lzy/.conda/envs/ltf/lib/python3.8/re.py", line 201, in search
    return _compile(pattern, flags).search(string)
  File "/home/lzy/.conda/envs/ltf/lib/python3.8/re.py", line 304, in _compile
    p = sre_compile.compile(pattern, flags)
  File "/home/lzy/.conda/envs/ltf/lib/python3.8/sre_compile.py", line 764, in compile
    p = sre_parse.parse(p, flags)
  File "/home/lzy/.conda/envs/ltf/lib/python3.8/sre_parse.py", line 948, in parse
    p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
  File "/home/lzy/.conda/envs/ltf/lib/python3.8/sre_parse.py", line 443, in _parse_sub
    itemsappend(_parse(source, state, verbose, nested + 1,
  File "/home/lzy/.conda/envs/ltf/lib/python3.8/sre_parse.py", line 836, in _parse
    raise source.error("missing ), unterminated subpattern",
re.error: missing ), unterminated subpattern at position 6








因为准确率在log日志中有一个汇总，因此这样每个准确率都被统计了两遍。






很棒，将这个汇总结果以如下形式输出：

```
Accuracy: [10%, 20%, ..., 共32个准确率] | 最大准确率/最大准确率的首次出现轮次
```





我有一个`config.env`，格式如下：

```
num_clients = 10
batch_size = 32
num_rounds = 32
epoch_client = 1
datasize_perclient = 32
datasize_valide = 1000
learning_rate = 0.001
ifPCA = False
ifCleanAnoma = True
PCA_rate = 1
PCA_nComponents = 2
attackList = []
attack_rate = 1
device = cuda:0
```

请读取其中的如下信息并输出：

+ epoch_client
+ learning_rate
+ batch_size
+ device






这是我融合了输出配置文件和输出准确率日志的代码。

```
def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 忽略空行和注释行
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config

def print_selected_config(config):
    keys_of_interest = ['epoch_client', 'learning_rate', 'batch_size', 'device']
    for key in keys_of_interest:
        if key in config:
            print(f"{key} = {config[key]}")
        else:
            print(f"{key} not found in the config file.")

if __name__ == "__main__":
    config_file_path = './result/2024.07.07-00:36:41/config.env'  # 将此替换为实际的配置文件路径
    config = read_config(config_file_path)
    print_selected_config(config)



import re

def extract_accuracies(log_file):
    accuracies = []
    with open(log_file, 'r') as file:
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

def print_summary(accuracies):
    accuracy_str = ", ".join([f"{acc}%" for acc in accuracies])
    max_accuracy, max_round = get_max_accuracy(accuracies)
    print(f"Accuracy: [{accuracy_str}] | 最大准确率: {max_accuracy}% / 最大准确率的首次出现轮次: {max_round}")

if __name__ == "__main__":
    log_file = './result/2024.07.07-00:36:41/stdout.txt'  # 将此替换为实际日志文件路径
    accuracies = extract_accuracies(log_file)
    print_summary(accuracies)
```

实际上，stdout.txt和config.env是在同一个文件中的。

现在我想请你修改这个代码，只需要填写一个文件夹路径，就能读取config.env中的配置信息和stdout.txt中的准确率信息。







太棒了。现在，我想以markdown表格的形式输出。






这不是我想要的markdown表格的形式。我想要的表格的形式是，将这些信息都输出到一行。

其中表头为：

| epoch_client | learning_rate | batch_size | batch_size | device | Accuracy | 最大准确率 | 最大准确率的首次出现轮次 |









太棒了，这就是我想要的结果。现在，我有多个文件夹，它们都在./result文件夹下：

```
./result/2024.07.07-00:36:41
./result/2024.07.07-00:41:37
...
./result/2024.07.07-04:14:49
```

它们都以日期时间的格式命名。我想让你提取`./result`文件夹中，日期从`2024.07.07-00:36:41`到`2024.07.07-04:14:49`的文件夹的配置信息和准确率信息。

请将它们提取出来，并汇总到一个markdown表格中。一个文件夹是表格的一行。







很棒。接下来将表头由

```
| epoch_client | learning_rate | batch_size | device | Accuracy | 最大准确率 | 最大准确率的首次出现轮次 |
```

修改为

```
| 单个客户端训练轮次 | learning rate | batch size | device | accuracy | 最大准确率 | 最大准确率的首次出现轮次 |
```





还记得我输出文件的样子吗？提取其中所有的日期时间，由最大的日期时间减去最小的日期时间得到程序执行耗时。并将这个结果作为一列添加到markdown表格中。










<!-- 在输出到markdown表格之前，请先对所有数据排序。排序规则依次为：

1. cuda:0的优先
1.1 若device为cuda:0，则按照learning_rate冲 -->






<!-- 在以下15行的输出结果中，混入了一行 -->






将输出结果按照如下方式排序后再输出到markdown表格中：

1. device字符串小的优先
2. epoch_client大的优先
3. 步长大的优先
4. batch_size大的优先





开始时间为`2024.07.07-00:41:37`，结束时间为`2024.07.07-04:14:49`





现在请再往markdown表格中添加一列。表头为`结果图`，内容为`./result/{日期文件夹名}/lossAndAccuracy.svg`





请为图片添加一个合理的markdown图片标签，而不是一个简单的图片路径字符串





accuracy单元格的内容太长了，能否为accuracy中的内容加一个横向滚动条，内容显示不完全时可以水平拖动以查看全部