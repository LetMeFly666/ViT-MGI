参数问题解决了。现在我想开始测试PCA有关的参数对于实验结果的影响，请你帮我挑选几个合适的参数来测试




PCA有哪些参数




解释一下“PCA降维后的主成分数目”





```
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_grads = pca.fit_transform(all_grads)
```

其中的`pca.fit_transform`可以重复使用吗？例如：

```
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_grads1 = pca.fit_transform(all_grads1)
reduced_grads2 = pca.fit_transform(all_grads2)
```

还是说我需要重新初始化一下`PCA`？

```
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)
reduced_grads1 = pca1.fit_transform(all_grads1)
pca2 = PCA(n_components=2)
reduced_grads2 = pca2.fit_transform(all_grads2)
```






我有一个`tempList`，格式如下：

```
tempList = [
    ((1, 2), 3),
    ((3, 2), 3),
    ...
]
```

对于其中的元素`a`（例如`a=((1, 2), 3)`，我想以如下规则排序：

首先`a[0][1]`小的优先，其次`a[0][0]`大的优先。

我应该怎么做？






帮我改写成sort(lambda)的形式