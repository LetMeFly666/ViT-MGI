<!--
 * @Author: LetMeFly
 * @Date: 2024-05-15 17:45:43
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-07-02 17:15:50
-->
# FLDefinder

联邦学习 ViT Backdoor防御的研究。

+ 进度地址：[人工智能 - 联邦学习(安全性) - 自用(ProjectDoing)](https://blog.letmefly.xyz/2024/01/06/Other-AI-FL-FederatedLearning-ProjectWritingIn1month/)
+ 分支[try0_poolAndExtra](https://github.com/LetMeFly666/FLDefinder/tree/try0_poolAndExtra)：因准确率太低，研究一半而Archive的分支
+ 分支[try1_changeFromPelta](https://github.com/LetMeFly666/FLDefinder/tree/try1_changeFromPelta)：在[Pelta](https://github.com/queyrusi/Pelta)的代码上修改，但其代码中似乎无FL相关部分，研究一半而Archive的分支

## Readme in Pelta

# Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments

Code base for the **Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments** paper (Queyrut, Schiavoni & Felber) accepted at ICDCS'23 (open access version soon available).

Code is provided for applying the Pelta defense scheme to an ensemble of Vision Transformer (ViT-L-16) and and Big Transfer Model (BiT-M-R101x3) against the Self-Attention Gradient Attack (original attack code from authors, [paper here](https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html)). The defense provided here works for CIFAR-10 and was coded entirely on PyTorch.
Parameters of the defense can be changed in the `env` file through the `PELTA` and `SHIELDED`parameters (set to `True` and `BOTH` by default).

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation Section (see below).</li>
  <li>Download the models from this Kaggle <a href="www.kaggle.com/reyacardov/ensemblemodels">dataset link</a>
  <li>Move both models into the ".\ExtendedPelta\Models" folder</li>
  <li>Run the main in the Python IDE of your choice</li>
</ol>

# Software Installation 

We use the following software packages: 
<ul>
  <li>pytorch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>numpy==1.19.2</li>
  <li>opencv-python==4.5.1.48</li>
  <li>python-dotenv==0.21.1</li>
</ul>

# System Requirements 

All our defenses were run on one 40GB A100 GPU and system RAM were of 16GB.

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

## End

### 参考文献

[^1]: Queyrut S, Schiavoni V, Felber P. Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments[C]//2023 IEEE 43rd International Conference on Distributed Computing Systems (ICDCS). IEEE, 2023: 626-637.
