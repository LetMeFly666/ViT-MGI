<!--
 * @Author: LetMeFly
 * @Date: 2024-07-05 21:09:55
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-07-15 15:17:07
-->
# 頑張って

emm，VsCode连接远程服务器的时候Latex插件的默认工作路径是```%WS1%```，然后就根本执行不了命令。

决定手动在终端make好了。还好```Latex```的插件能在PDF发生变化时自动重载。

诶，可以在```{工作目录}/.vscode/settings.json```中添加```"latex-workshop.latex.autoBuild.run": "never",```，这样Latex拓展不会在保存时被激活。然后下载插件```Run on Save```，添加如下参数，就完美了。

```json
"emeraldwalk.runonsave": {
    "commands": [
        {
            "match": "/home/lzy/ltf/Codes/FLDefinder/paper/main.tex",
            // "cmd": "cd /home/lzy/ltf/Codes/FLDefinder/paper/ && /usr/local/texlive/2024/bin/x86_64-linux/xelatex main.tex"
            "cmd": "cd /home/lzy/ltf/Codes/FLDefinder/paper/ && make"
        },
    ]
}
```

TODO: un ignore PDF


** Conference Paper **
Before submitting the final camera ready copy, remember to:

 1. Manually equalize the lengths of two columns on the last page
 of your paper;

 2. Ensure that any PostScript and/or PDF output post-processing
 uses only Type 1 fonts and that every step in the generation
 process uses the appropriate paper size.

To change:

<!-- 0. 再引几篇 -->
<!-- 1. PCA介绍那些太单薄了 -->
2. 图可以压缩，每张图保持一样格式
<!-- 3. contribution  合并合并 -->
4. 什么格式都应该统一
5. 图 grid(False)
<!-- 6. 特征层提取的算法移动到METHODOLOGY -->