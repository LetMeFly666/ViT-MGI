# FLDefinder

[Tisfy](https://github.com/LetMeFly666)和[Wang Bo](https://github.com/Pesuking)的论文尝试（联邦学习(FL)的攻击和防御问题）

目前正在进行

<details>
<summary><del>我们的进度</del></summary>

进度地址：[人工智能 - 联邦学习(安全性) - 自用(ProjectDoing)](https://blog.letmefly.xyz/2024/01/06/Other-AI-FL-FederatedLearning-ProjectWritingIn1month/)

</details>

## Log

### 001

2024.5.9上午-2024.5.10上午，cifar10对于grad_ascent的5个防御方式

<details><summary>参数配置</summary>

```python
# 基础配置
dataset_name = args.dataset_name
Ph = 15  # 客户端数量
num_iter = 50   # 总epoch数
local_epoch = 2  # 每个客户端的local_epoch
participant_factor = 0.7  # 每轮训练的参与者所占比例
loader_batch_size = 500   # 数据加载器的batch_size（一次从loader中会获得多少数据）

# 攻击相关配置
attack_mode = args.attack_mode
malicious_factor = 0.3  # 恶意客户端的所占比例
scale_target = 0
start_attack = 20


# 防御相关的配置
defend_mode = args.defend_mode
layers_to_look = ["patch_embed", "attn", "mlp"] # ['patch_embed', 'mlp']
kernel = 40
k_nearest = int(Ph * participant_factor) // 2 + 1
accs_mean_list = []
accs_std_list = []
slot = 4
distance = 0.01
```

</details>

两张3090，内存占用约```23800/24260```，耗时约24h（略大于）。

**结果**

不防御的状态下（defend0），攻击未开始时的准确率已经明显大于其他其他4种。防御有效果但似乎不是很明显。不防御的情况下攻击开始时准确率略微下降，后续也再次回升。

**TODO**

+ 新的运行结果会覆盖旧的相同配置的运行结果，不如每次运行都新建一个文件夹，文件夹下新建一个配置文件说明本次的运行时间及其他一些配置信息。
+ 可能需要一种有效的方式来衡量防御结果是否有意义，例如绘图？
+ 有没有必要控制一下变量，例如前20轮只跑一次，都在此基础上进行后续攻防[^1]；或者使用相同的数据集（rand出来后选一样的）。
+ 是否与选层```layers_to_look```有关。

## End

### 参考文献

[^1]: Shen L, Zhang Y, Wang J, et al. Better together: Attaining the triad of Byzantine-robust federated learning via local update amplification[C]//Proceedings of the 38th Annual Computer Security Applications Conference. 2022: 201-213.
