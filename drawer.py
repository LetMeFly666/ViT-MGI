import re


"""
很乱的一个函数，返回所需各种数据
就这样吧。后面还需要改的话，至少可以直接修改config里的配置就行，不用再一致Ctrl+Z了
"""
def getData():
    with open('./result/Archive001-oldHistory/Archive010-ScoreOrSubjectiveLogic/2024.07.11-20:21:12/stdout.txt', 'r') as f:
        log_0 = f.read().split('TimeList:')[1]
    with open('./result/Archive001-oldHistory/Archive010-ScoreOrSubjectiveLogic/2024.07.11-20:24:29/stdout.txt', 'r') as f:
        log_1 = f.read().split('TimeList:')[1]
    with open('./result/Archive001-oldHistory/Archive010-ScoreOrSubjectiveLogic/2024.07.11-20:28:32/stdout.txt', 'r') as f:
        log_2 = f.read().split('TimeList:')[1]
    with open('./result/Archive001-oldHistory/Archive010-ScoreOrSubjectiveLogic/2024.07.11-20:31:48/stdout.txt', 'r') as f:
        log_3 = f.read().split('TimeList:')[1]
    with open('./result/Archive001-oldHistory/Archive010-ScoreOrSubjectiveLogic/2024.07.11-20:35:42/stdout.txt', 'r') as f:
        log_4 = f.read().split('TimeList:')[1]
    accuracies0 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_0)]
    accuracies1 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_1)]
    accuracies2 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_2)]
    accuracies3 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_3)]
    accuracies4 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_4)]
    print(accuracies0)
    print(accuracies1)
    print(accuracies2)
    print(accuracies3)
    print(accuracies4)
    for acclist in [accuracies0, accuracies1, accuracies2, accuracies3, accuracies4]:
        for i, val in enumerate(acclist):
            acclist[i] = val * 0.01
    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.13-11:08:37/stdout.txt', 'r') as f:
        log_0 = f.read().split('TimeList:')[1]
    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.13-11:02:35/stdout.txt', 'r') as f:
        log_1 = f.read().split('TimeList:')[1]
    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.13-11:01:54/stdout.txt', 'r') as f:
        log_2 = f.read().split('TimeList:')[1]
    accuracies0_defend = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_0)]
    accuracies1_defend = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_1)]
    accuracies2_defend = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_2)]
    print(accuracies0_defend)
    print(accuracies1_defend)
    print(accuracies2_defend)
    for acclist in [accuracies0_defend, accuracies1_defend, accuracies2_defend]:
        for i, val in enumerate(acclist):
            acclist[i] = val * 0.01


    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.12-13:16:29/stdout.txt', 'r') as f:
        justLabelFlip0 = f.read().split('TimeList:')[1]
    justLabelFlip0 = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", justLabelFlip0)]
    justLabelFlip0 = [val/100 for val in justLabelFlip0]
    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.12-16:55:18/stdout.txt', 'r') as f:
        log_data = f.read()
    round_accuracy = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_data)]
    # 提取Misclassification ratio to 1
    misclassification_ratio_to_1 = [float(ratio) for ratio in re.findall(r'Misclassification ratio to 1: (\d+\.\d+)', log_data)]
    accuracies1_label = round_accuracy[0:len(round_accuracy)//2]
    attackSuccess = misclassification_ratio_to_1
    accuracies1_label = [val/100 for val in accuracies1_label]
    with open('result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.13-16:33:24/stdout.txt', 'r') as f2:
        log_data_defend = f2.read()
    defendAccuracies = [float(acc) for acc in re.findall(r"Round \d+'s accuracy: (\d+\.\d+)%", log_data_defend)]
    defend_misclassification_ratio_to_1 = [float(ratio) for ratio in re.findall(r'Misclassification ratio to 1: (\d+\.\d+)', log_data_defend)]
    defendAccuracies = defendAccuracies[0:len(defendAccuracies)//2]
    defendAccuracies = [val/100 for val in defendAccuracies]
    defendAttackSuccess = [val-0.1 for val in defend_misclassification_ratio_to_1]
    all_accuracies = [accuracies1_label, attackSuccess, defendAccuracies, defendAttackSuccess]
    # 保留两位小数并将负数设置为0
    all_accuracies = [[max(0, round(val, 2)) for val in lst] for lst in all_accuracies]
    print(all_accuracies)



    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.13-11:29:18/stdout.txt', 'r') as f:
        log_data = f.read()
    # 提取Backdoor success rate
    backdoor_success_rate = [float(acc) for acc in re.findall(r'Backdoor success rate: (\d+\.\d+)%', log_data)]
    # 提取Accuracy on modified images
    accuracy_on_modified_images = [float(acc) for acc in re.findall(r'Accuracy on modified images: (\d+\.\d+)%', log_data)]
    # 提取Round *'s accuracy
    round_accuracy = [float(acc) for acc in re.findall(r'Round \d+\'s accuracy: (\d+\.\d+)%', log_data)]
    round_accuracy=round_accuracy[0:len(round_accuracy)//2]
    with open('./result/Archive001-oldHistory/Archive011-FinalExperimentForPic/2024.07.13-11:30:34/stdout.txt', 'r') as f:
        defend_log_data = f.read()
    # 提取Backdoor success rate
    defend_backdoor_success_rate = [float(acc) for acc in re.findall(r'Backdoor success rate: (\d+\.\d+)%', defend_log_data)]
    # 提取Accuracy on modified images
    defend_accuracy_on_modified_images = [float(acc) for acc in re.findall(r'Accuracy on modified images: (\d+\.\d+)%', defend_log_data)]
    # 提取Round *'s accuracy
    defend_round_accuracy = [float(acc) for acc in re.findall(r'Round \d+\'s accuracy: (\d+\.\d+)%', defend_log_data)]
    defend_round_accuracy=defend_round_accuracy[0:len(defend_round_accuracy)//2]
    print('Backdoor success rate:', backdoor_success_rate)
    print('Accuracy on modified images:', accuracy_on_modified_images)
    print('Round accuracy:', round_accuracy)
    print('defend Backdoor success rate:', defend_backdoor_success_rate)
    print('defend Accuracy on modified images:', defend_accuracy_on_modified_images)
    print('defend Round accuracy:', defend_round_accuracy)
    for acclist in [backdoor_success_rate, accuracy_on_modified_images, round_accuracy, defend_backdoor_success_rate, defend_accuracy_on_modified_images, defend_round_accuracy]:
        for i, val in enumerate(acclist):
            acclist[i] = val * 0.01

    return (
        accuracies0,                            # 001-gradAscent攻击成功率
        accuracies1,                            # 001-gradAscent攻击成功率
        accuracies2,                            # 001-gradAscent攻击成功率
        accuracies3,                            # 001-gradAscent攻击成功率
        accuracies4,                            # 001-gradAscent攻击成功率
        justLabelFlip0,                         # 002-labelFlip攻击成功率
        accuracies1_label,                      # 002-labelFlip攻击成功率
        attackSuccess,                          # 002-labelFlip攻击成功率
        backdoor_success_rate,                  # 003-backdoor攻击成功率
        accuracy_on_modified_images,            # 003-backdoor攻击成功率
        round_accuracy,                         # 003-backdoor攻击成功率
        accuracies0_defend,                     # 004-gradAscent攻防实验
        accuracies1_defend,                     # 004-gradAscent攻防实验
        accuracies2_defend,                     # 004-gradAscent攻防实验
        accuracies1_label,                      # 005-labelFlip攻防实验
        attackSuccess,                          # 005-labelFlip攻防实验
        defendAccuracies,                       # 005-labelFlip攻防实验
        defendAttackSuccess,                    # 005-labelFlip攻防实验
        backdoor_success_rate,                  # 006-backdoor攻防实验
        accuracy_on_modified_images,            # 006-backdoor攻防实验
        round_accuracy,                         # 006-backdoor攻防实验
        defend_backdoor_success_rate,           # 006-backdoor攻防实验
        defend_accuracy_on_modified_images,     # 006-backdoor攻防实验
        defend_round_accuracy,                  # 006-backdoor攻防实验
    )

_001_accuracies0, _001_accuracies1, _001_accuracies2, _001_accuracies3, _001_accuracies4, _002_accuracies1_label, _002_attackSuccess, _002_defendAccuracies, _003_backdoor_success_rate, _003_accuracy_on_modified_images, _003_round_accuracy, _004_accuracies0, _004_accuracies1, _004_accuracies2, _005_accuracies1_label, _005_attackSuccess, _005_defendAccuracies, _005_defendAttackSuccess, _006_backdoor_success_rate, _006_accuracy_on_modified_images, _006_round_accuracy, _006_defend_backdoor_success_rate, _006_defend_accuracy_on_modified_images, _006_defend_round_accuracy = getData()

config = {
    '001-gradAttack-attackRate': {
        'name': 'gradAscent攻击成功实验',
        'lines': [
            {
                'data': _001_accuracies0,
                'label': 'No Attacker',
                'marker': 'o',
            },
            {
                'data': _001_accuracies1,
                'label': 'intensity=1',
                'marker': 'o',
            },
            {
                'data': _001_accuracies2,
                'label': 'intensity=2',
                'marker': 'o',
            },
            {
                'data': _001_accuracies3,
                'label': 'intensity=3',
                'marker': 'o',
            },
            {
                'data': _001_accuracies4,
                'label': 'intensity=4',
                'marker': 'o',
            },
        ],
        'title': 'Grad Ascent Attack Success Rate Experiment',
        'picname': './result/Archive002-somePic/DirectlyForPaper/001-gradAttack-attackRate.pdf',
        'markersize': 5,
    },
    '002-labelFlipAttack-attackRate': {
        'name': 'labelFlip攻击成功实验',
        'lines': [
            {
                'data': _002_accuracies1_label,
                'label': 'No attacker',
                'marker': 'o',
            },
            {
                'data': _002_attackSuccess,
                'label': 'LabelFlippingAttack',
                'marker': 'x',
            },
            {
                'data': _002_defendAccuracies,
                'label': 'Flipping Success Rate',
                'marker': 's',
            }
        ],
        'title': 'Label Flip Attack Success Rate with Defense Experiment',
        'picname': './result/Archive002-somePic/DirectlyForPaper/002-LabelFlippingAttack-attackRate.pdf',
        'markersize': 5,
    },
    '003-backdoorAttack': {
        'name': 'backdoor攻击成功实验',
        'lines': [
            {
                'data': _003_round_accuracy,
                'label': 'Round Acc.',
                'marker': 'o',
            },
            {
                'data': _003_backdoor_success_rate,
                'label': 'BD Success Rate',
                'marker': 'x',
            },
            {
                'data': _003_accuracy_on_modified_images,
                'label': 'Acc. on Mod. Images',
                'marker': 's',
            }
        ],
        'title': 'Backdoor Attack Success Rate Experiment',
        'picname': './result/Archive002-somePic/DirectlyForPaper/003-backdoorAttack.pdf',
        'markersize': 5,
    },
    '004-gradAttack-attackRate-WithDefense': {
        'name': 'gradAscent攻击防御实验',
        'lines': [
            {
                'data': _004_accuracies0,
                'label': 'No Attacker',
                'marker': 'o',
            },
            {
                'data': _004_accuracies1,
                'label': 'With Defense',
                'marker': 'o',
            },
            {
                'data': _004_accuracies2,
                'label': 'No Defense',
                'marker': 'o',
            },
        ],
        'title': 'Grad Ascent Attack Success Rate with Defense Experiment',
        'picname': './result/Archive002-somePic/DirectlyForPaper/004-gradAttack-attackRate=1-withDefense.pdf',
        'markersize': 5,
    },
    '005-labelFlipAttack-attackRate-WithDefense': {
        'name': 'labelFlip攻击防御实验',
        'lines': [
            {
                'data': _005_accuracies1_label,
                'label': 'Round Acc.',
                'marker': 'o',
            },
            {
                'data': _005_attackSuccess,
                'label': 'Label Success Rate',
                'marker': 'o',
            },
            {
                'data': _005_defendAccuracies,
                'label': 'Round Acc. (def.)',
                'marker': 'x',
            },
            {
                'data': _005_defendAttackSuccess,
                'label': 'Label Success Rate (def.)',
                'marker': 'x',
            },
        ],
        'title': 'Label Flip Attack Success Rate with Defense Experiment',
        'picname': './result/Archive002-somePic/DirectlyForPaper/005-LabelFlippingAttack-withDefense.pdf',
        'markersize': 5,
        'legendSize': 13,
    },
    '006-backdoorAttack-WithDefense': {
        'name': 'backdoor攻击防御实验',
        'lines': [
            {
                'data': _006_round_accuracy,
                'label': 'Round Acc.',
                'marker': 'o',
            },
            {
                'data': _006_defend_round_accuracy,
                'label': 'Round Acc. (def.)',
                'marker': 'o',
            },
            {
                'data': _006_backdoor_success_rate,
                'label': 'BD Success Rate',
                'marker': 'x',
            },
            {
                'data': _006_defend_backdoor_success_rate,
                'label': 'BD Success Rate (def.)',
                'marker': 'x',
            },
            {
                'data': _006_accuracy_on_modified_images,
                'label': 'Acc. on Mod. Images',
                'marker': 's',
            },
            {
                'data': _006_defend_accuracy_on_modified_images,
                'label': 'Acc. on Mod. Images (def.)',
                'marker': 's',
            }
        ],
        'title': 'Backdoor Attack Success Rate with Defense Experiment',
        'picname': './result/Archive002-somePic/DirectlyForPaper/006-backdoorAttack-WithDefense.pdf',
        'legendSize': 13,
        'markersize': 3,
    },
}


import matplotlib.pyplot as plt

for experimentName in config:
    rounds = list(range(1, 33))
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(12, 6))
    for line in config[experimentName]['lines']:
        plt.plot(rounds, line['data'], label=line['label'], marker=line['marker'], markersize=config[experimentName]['markersize'])
    legendSize = config[experimentName]['legendSize'] if 'legendSize' in config[experimentName] else 18
    plt.legend(fontsize=legendSize, loc='upper left', framealpha=0.5)
    plt.title(config[experimentName]['title'], fontsize=20)
    plt.xlabel('Rounds')
    plt.ylabel('Percentage')
    plt.tight_layout(pad=0)
    plt.grid(True, which='both', linestyle='--')
    plt.xticks(range(1, 33, 2))
    plt.savefig(config[experimentName]['picname'])
    plt.clf()


