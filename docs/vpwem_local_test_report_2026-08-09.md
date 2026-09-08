# VPWEM 本机测试环境与结果

日期：2026-08-09  
工作目录：`/data/wjr/vpwem/code_vpwem`

## 结论

本机可以运行 VPWEM 单元测试。历史 Robomimic 运行日志和 W&B metadata 均指向以下解释器：

```text
/home/wjr/miniconda3/envs/vpwem5090/bin/python
```

该环境检测结果：

```text
Python: 3.9.25
PyTorch: 2.8.0+cu128
CUDA runtime reported by PyTorch: 12.8
GPU: NVIDIA GeForce RTX 5090 D v2
NumPy: 1.22.4
Matplotlib: 3.5.0
pytest: 8.4.1
Robomimic: 0.2.0
```

## 测试命令

目标测试：

```bash
cd /data/wjr/vpwem/code_vpwem
PYTHONPATH=. /home/wjr/miniconda3/envs/vpwem5090/bin/pytest -q \
  tests/test_per_memory.py \
  tests/test_memory_interventions.py
```

结果：`11 passed, 0 failed`。

全量测试：

```bash
cd /data/wjr/vpwem/code_vpwem
PYTHONPATH=. /home/wjr/miniconda3/envs/vpwem5090/bin/pytest -q tests
```

结果：`21 passed, 0 failed`。另有 6 条第三方依赖弃用警告，不影响本次通过状态。

## 环境定位证据

- `train_longsquare_vpwem_seed100.log` 的异常栈来自 `vpwem5090/lib/python3.9/site-packages`。
- `artifacts/per_vpwem/formal_eval` 下的 W&B metadata 将 executable 记录为 `/home/wjr/miniconda3/envs/vpwem5090/bin/python`。
- 本机已有 Robomimic checkpoint、评估结果和 PER/FIFO 运行产物，不需要仅为单元测试切换到服务器。

## 不应使用的环境

`/home/wjr/miniconda3/envs/vpwem` 当前含 NumPy 2.0.2 和按 NumPy 1.x 编译的 Matplotlib 3.5.0。导入 Matplotlib 时出现二进制 ABI 错误，pytest 会在收集阶段停止。为保护旧实验环境，本次没有修改其包版本。

项目根目录的 `.venv` 只安装了 pytest，没有安装完整 VPWEM 依赖，因此也不是模型测试环境。

## 结果边界

单元测试通过证明当前被覆盖的张量形状、有界事件库、在线 reset、辅助损失接入和三类记忆干预符合测试断言。它不证明训练可复现、成功率正确、PER 优于基线或记忆被策略因果使用；这些结论仍需 checkpoint 评估和干预实验支持。
