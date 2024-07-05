<!--
 * @Author: LetMeFly
 * @Date: 2024-05-15 17:45:43
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-07-06 00:32:19
-->
# FLDefinder

联邦学习 ViT Backdoor防御的研究。

+ 进度地址：[人工智能 - 联邦学习(安全性) - 自用(ProjectDoing)](https://blog.letmefly.xyz/2024/01/06/Other-AI-FL-FederatedLearning-ProjectWritingIn1month/)
+ 分支[try0_poolAndExtra](https://github.com/LetMeFly666/FLDefinder/tree/try0_poolAndExtra)：因准确率太低，研究一半而Archive的分支
+ 分支[try1_changeFromPelta](https://github.com/LetMeFly666/FLDefinder/tree/try1_changeFromPelta)：在[Pelta](https://github.com/queyrusi/Pelta)的代码上修改，但其代码中似乎无FL相关部分，研究一半而Archive的分支

## Log

### Log001 - 2024.5.14-2024.5.19

暂时停止在[原有](https://github.com/LetMeFly666/FLDefinder/commit/c830b55950ba84a8dd657bbd4ecfa247c6c3e8a5)基础上继续更改，开始寻找现有的联邦学习ViT Backdoor的代码并在此基础上进行更改。

+ <del>搜索关键词：<code>("ViT" OR "Vision Transformer") AND "Backdoor" AND ("Federated Learning" OR "FL") AND "github.com"</code>（这样Sensitivity也会被检索上）</del>
+ 搜索关键词：```"Vision Transformer" AND "Backdoor" AND ("Federated Learning" OR "FL") AND "github.com"```

确认文章[^1]。下载其[代码](https://github.com/queyrusi/Pelta)与[数据集](https://www.kaggle.com/datasets/reyacardov/ensemblemodels)尝试开始运行。

### Log003 - 2024.5.19-2024.5.23

1. 修改代码文件结构成功跑通
2. 重命名```env```文件为更加通用（标准）的```.env```
3. 实现了自定义的```print```函数，在调用```initPrint```函数后，以后的所有```print```都会在原来的基础上同时往initPrint时的文件中输出一份。

但是此时调用的一些库的Warning不是调用print函数显示到终端的，就无法同时悄悄地写入到文件中一份。

### Log004 - 2024.7.2_17:00-2024.7.2_23:50

1. 先将“FL”、“ViT”的代码跑通，首先拥有一个能在半小时内训练出大约90%多准确率的ViT联邦学习框架。

先支持大约5个客户端即可。

实际上到```2024.7.3 23:59```左右才实现。

### Log005 - 2024.7.4_9:00-2024.7.4_23:59

1. 优化模型（包括调整learning rate，优化数据分发方式，调整训练集大小等）。结果：lr```0.001略优于0.0025优于0.01```且```0.001略优于0.0005```。基本可以确定```0.001```是一个比较合适的值。
2. 融入攻防。

今晚走之前跑上两个长训练。

+ lr=0.001，epoch=50x3，maxAcc=57.37%
+ lr=0.005，epoch=50x3，maxAcc=54.03%
+ ~~lr=0.01，epoch=60x1，maxAcc=48.17%~~
+ lr=0.01，epoch=200x3，maxAcc=60.13%
+ lr=0.01，epoch=50x3，maxAcc=51.37%
+ lr=0.02，epoch=200x3，Adam+StepLR，maxAcc=29.60%（后面想起来每次下发模型优化器都会重置）

### Log006 - 2024.7.5_9:00-2024.7.5_11:00

暂不使用非预训练模型，先使用预训练模型，将参数调整到一个不错的状态。

| lr     | epoch | dataPerEpoch | maxAcc  | timeConsume | 备注           |
|--------|-------|--------------|---------|-------------|----------------|
| 0.001  | 30x1  | 10x32        | 96.9%   | 165s        |                |
| 0.0001 | 30x1  | 10x32        | 95.8%   | 164s        |                |
| 0.0001 | 60x1  | 10x32        | 97.6%   | 319s        |                |
| 0.0001 | 150x1 | 10x32        | 98.8%   | 790s        | 116轮首次达到  |
| 0.001  | 150x1 | 10x32        | 98.9%   | 808s        | 71轮首次达到   |
| 0.001  | 1300x1| 10x32        | 99.5%   | 7099s       | 1141轮首次达到 |
| 0.02   | 1300x1| 10x32        | 99.7%   | 7433s       | 570轮首次达到  |

其中：

+ lr：步长（学习率）
+ epoch=30x1：服务器主持训练30轮，每轮每个客户端训练1轮
+ dataPerEpoch=10x32：10个客户端，每个客户端每次训练下发32个训练数据

### Log007 - 2024.7.5_13:30-2024.7.5_18:00

+ 成功融入最基础的梯度上升攻击与PCA检测的防御

使用的是内置的PCA算法，PCA偏离倍数为```0.1```和```0.3```时都不如```0.2```时。因此选择偏离倍数为```0.2```

当恶意用户的攻击方式是梯度乘以-2时，几乎都能正常识别；当梯度乘以-1时，识别准确率为```68.75%```。而前10轮次的识别准确率为```90%```。这可能是由“接近饱和时梯度变化变小”导致的。

**但是：** PCA直接计算十个客户端的所有参数（1×n）太慢了。

然后我把攻击者设置为2/10个，PCA就寄了。攻击力度为1的话（梯度×-1）基本上是在瞎输出，攻击力度为2的话基本上每次能在两个里面抓到一个。

| 攻击者 | 攻击力度 | PCA的偏离倍数 | 表现                        |
|--------|----------|---------------|-----------------------------|
| 2/10   | 1        | 2             | 基本上在瞎输出              |
| 2/10   | 2        | 2             | 基本上每次能在两个里面抓到一个 |
| 2/10   | 2        | 1             | 32次中有31次完全正确，另外一次多抓了一个 |
| 3/10   | 2        | 1             | 32次中有8次完全正确，21次少抓了一个，3次少抓了两个|
| 3/10   | 1        | 1             | 32次中有10次完全正确，9次少抓了一个，2次少抓了两个<br/>1次少一多一，5次少两且多一，5次少三多一|

### Log008 - 2024.7.5_20:40-2024.7.6_0:30

+ 先写上题目和摘要（要做什么事情）
+ 之后是Intro和Related Work

结果刚配置好了VsCode、overleaf(byr)、中文: [c3db3c](https://github.com/LetMeFly666/FLDefinder/tree/c3db3c6865d9a9b1cac8016fe2fadb6a3c2fbc05)

### Log009 - 2024.7.6_9:10-2024.7.6_10:30

完成Log008中未完成的内容。

### TODO

- [x] 每个客户端下次数据会发生变化
- [x] 客户端本地训练多个（例如3）epoch
- [ ] 先写个题目和摘要
- [ ] IF零信任？
- [ ] 了解一些攻防手段，例如主成分萃取/最大池化及其关系
- [ ] 不上传客户端的diff，而是直接上传所有客户端的参数然后在服务端求平均
- [ ] 损失函数求模型总的损失函数

## End

The End.
