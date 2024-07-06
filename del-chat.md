这是我的论文引言（暂定），请你据此提出几个关键词。

```
联邦学习可以在数据不离开客户端的前提下进行模型训练，保护了用户的隐私且减小了中央服务器的压力。然而，在联邦学习的过程中，经常会有恶意客户端的存在。当然，恶意用户的数量一般会占据较小的部分。本文对联邦学习过程中用户上传上来的梯度变化进行分析，首先利用最大池化技术提取主要特征的同时降低，利用主成分分析(PCA)算法\textcolor{red}{（这里还有待添加更多的算法）}对恶意用户进行识别。单次被识别为恶意用户可能是由于误判，因此本文使用主观逻辑模型来对用户的信用等级进行评估，并依据用户的可信度来调整模型聚合时的权重。

结果表明，当前方法对于恶意用户的识别率、恶意用户的识别效率，以及程序自身的鲁棒性都有所提升。我们将其命名为FLDefinder\textcolor{red}{（名字待定）}并发布了其源代码，以方便该领域的未来研究：\href{https://github.com/LetMeFly666/FLDefinder}{https://github.com/LetMeFly666/FLDefinder}\textcolor{red}{（论文投稿前此仓库为Private状态不可访问）}。
```




请使用专业的英文词汇






latex论文引用






解释如下latex代码

```
\begin{thebibliography}{00}
\bibitem{b1} G. Eason, B. Noble, and I. N. Sneddon, ``On certain integrals of Lipschitz-Hankel type involving products of Bessel functions,'' Phil. Trans. Roy. Soc. London, vol. A247, pp. 529--551, April 1955.
\bibitem{b2} J. Clerk Maxwell, A Treatise on Electricity and Magnetism, 3rd ed., vol. 2. Oxford: Clarendon, 1892, pp.68--73.
\bibitem{b3} I. S. Jacobs and C. P. Bean, ``Fine particles, thin films and exchange anisotropy,'' in Magnetism, vol. III, G. T. Rado and H. Suhl, Eds. New York: Academic, 1963, pp. 271--350.
\bibitem{b4} K. Elissa, ``Title of paper if known,'' unpublished.
\bibitem{b5} R. Nicole, ``Title of paper with only first word capitalized,'' J. Name Stand. Abbrev., in press.
\bibitem{b6} Y. Yorozu, M. Hirano, K. Oka, and Y. Tagawa, ``Electron spectroscopy studies on magneto-optical media and plastic substrate interface,'' IEEE Transl. J. Magn. Japan, vol. 2, pp. 740--741, August 1987 [Digests 9th Annual Conf. Magnetics Japan, p. 301, 1982].
\bibitem{b7} M. Young, The Technical Writer's Handbook. Mill Valley, CA: University Science, 1989.
\end{thebibliography}
```




打开谷歌学术搜索一篇论文时，会有“引用”选项。

引用时有多种引用格式，我应该选择哪一种？





如何使用更加灵活的`bibitem`？






Package biblatex Warning: Please (re)run Biber on the file:
(biblatex)                main
(biblatex)                and rerun LaTeX afterwards.






我使用的是xelatex  请给我所有要执行的命令





我在编译生成的PDF里看到参考文献了，但是格式好怪。如何修改成这样的格式

```
[1] G. Eason, B. Noble, and I. N. Sneddon, “On certain integrals of
Lipschitz-Hankel type involving products of Bessel functions,”
Phil. Trans. Roy. Soc. London, vol. A247, pp. 529–551, April
1955.
[2] J. Clerk Maxwell, A Treatise on Electricity and Magnetism, 3rd
ed., vol. 2. Oxford: Clarendon, 1892, pp.68–73.
[3] I. S. Jacobs and C. P. Bean, “Fine particles, thin
```




现在很像了，但是为什么里面还有下划线？