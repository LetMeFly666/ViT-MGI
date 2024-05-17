# FLDefinder

联邦学习 ViT Backdoor防御的研究。

+ 进度地址：[人工智能 - 联邦学习(安全性) - 自用(ProjectDoing)](https://blog.letmefly.xyz/2024/01/06/Other-AI-FL-FederatedLearning-ProjectWritingIn1month/)
+ 分支[try0_poolAndExtra](https://github.com/LetMeFly666/FLDefinder/tree/try0_poolAndExtra)：因准确率太低，研究一半而Archive的分支

## Log

### Log001 - 2024.5.14-2024.5.19

暂时停止在[原有](https://github.com/LetMeFly666/FLDefinder/commit/c830b55950ba84a8dd657bbd4ecfa247c6c3e8a5)基础上继续更改，开始寻找现有的联邦学习ViT Backdoor的代码并在此基础上进行更改。

+ <del>搜索关键词：<code>("ViT" OR "Vision Transformer") AND "Backdoor" AND ("Federated Learning" OR "FL") AND "github.com"</code>（这样Sensitivity也会被检索上）</del>
+ 搜索关键词：```"Vision Transformer" AND "Backdoor" AND ("Federated Learning" OR "FL") AND "github.com"```

确认文章[^1]。下载其[代码](https://github.com/queyrusi/Pelta)与[数据集](https://www.kaggle.com/datasets/reyacardov/ensemblemodels)尝试开始运行。

## End

### 参考文献

[^1]: Queyrut S, Schiavoni V, Felber P. Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments[C]//2023 IEEE 43rd International Conference on Distributed Computing Systems (ICDCS). IEEE, 2023: 626-637.
