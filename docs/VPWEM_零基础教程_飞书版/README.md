# VPWEM 零基础学习文件库

这是完全保存在本机仓库里的离线教程，不依赖飞书。

## 推荐打开方式

在仓库根目录执行：

```bash
bash docs/VPWEM_零基础教程_飞书版/打开教程.sh
```

然后在浏览器访问：

```text
http://127.0.0.1:8765/index.html
```

`index.html` 是包含全部章节和图片的单文件版本。也可以直接用浏览器打开：

```text
/home/wjr/code/vpwem/docs/VPWEM_零基础教程_飞书版/index.html
```

## 文件目录

```text
00_阅读地图与学习目标.md
01_先补地基_张量到QKV.md
02_一张图读懂VPWEM总架构.md
03_感知器与QFormer记忆层.md
04_动作Transformer与Diffusion.md
05_训练与在线推理调用链.md
06_如何修改网络结构并保证改对.md
index.html                 全部章节合并后的离线网页
教程样式.css               网页样式
打开教程.sh                启动本地网页服务
```

Markdown 是可编辑源文件；修改后可重新运行：

```bash
bash docs/VPWEM_零基础教程_飞书版/重新生成网页.sh
```
