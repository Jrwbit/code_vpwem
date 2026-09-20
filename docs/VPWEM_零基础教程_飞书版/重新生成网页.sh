#!/usr/bin/env bash
set -euo pipefail

LIBRARY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LIBRARY_DIR"

pandoc \
  00_阅读地图与学习目标.md \
  01_先补地基_张量到QKV.md \
  02_一张图读懂VPWEM总架构.md \
  03_感知器与QFormer记忆层.md \
  04_动作Transformer与Diffusion.md \
  05_训练与在线推理调用链.md \
  06_如何修改网络结构并保证改对.md \
  07_原始图像到VPWEM动作全流程图.md \
  --from=gfm \
  --to=html5 \
  --standalone \
  --toc \
  --toc-depth=3 \
  --metadata title="VPWEM 零基础学习文件库" \
  --css=教程样式.css \
  --embed-resources \
  --resource-path="$LIBRARY_DIR:$LIBRARY_DIR/.." \
  --output=index.html

echo "已生成：$LIBRARY_DIR/index.html"
