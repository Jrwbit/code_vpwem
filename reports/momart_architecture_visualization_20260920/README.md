# MoMaRT 交互式网络结构浏览器

打开 `index.html` 即可离线使用，无需启动服务器或联网。

页面包含：

- 六种方法切换：DP、DP-PTP、No-History、Direct、Direct-token2、VPWEM；
- 从实际 `nn.Module` 自动递归生成的可折叠模块树；
- 每个模块的递归参数量、完整模块路径；
- dummy forward 捕获的实际输入/输出 tensor shape；
- 从模型 `memory_mask` 直接生成的 cross-attention mask；
- 实验入口、协议和总参数量。

数据文件 `architectures.json` 可用于其他前端或分析脚本。`preview.png` 是首页预览。

重新导出：

```bash
PYTHONPATH=. /home/wjr/miniconda3/envs/vpwem/bin/python \
  tools/export_momart_architecture_html.py
```
