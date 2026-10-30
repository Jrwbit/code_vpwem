# Boxed No-History on BHR

本实验在归档 No-History 的当前观测上画目标检测框。DP、观察编码器、任务 embedding 和动作损失均直接使用 momart_no_history_baseline.model；没有历史库、memory token、ROI 分支或 bbox 数值条件。

## 输入规则

所有输入的主视角 `rgb` 最近两帧都在原始 90×120 RGB 上画红色 [255,0,0]、1 像素边框。模型输入中不画文字、分数或视频标题。未检测到碗时保留原图。多候选时使用检测器选出的最高置信候选。腕部 RGB、两个深度流、scan、proprio 和动作标签不变；原 baseline 的随机裁剪、归一化仍在画框之后执行，裁剪可能截去部分框。

训练从完成的 RGB-only Grounding DINO bbox sidecar 读取框，按帧画到图像副本，不改写原始 Zarr/HDF5。在线评测从环境返回的最近两帧 RGB 运行同一冻结检测器再画框；评测启动时校验检测器 metadata 与训练完全一致。检测耗时计入推理时间。检测器只读主视角 RGB，使用 red bowl 提示和红碗颜色先验，不使用模拟器物体真值。在线评测的视频录制保留原环境画面，框仅在模型输入路径绘制。

## 训练与评测

独立输出：/home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100。
固定 physical/effective BS256，150000 个 optimizer updates，每 25000 保存模型（含 EMA、优化器）与 scheduler，全部保留。从头训练，不加载之前 smoke 或其他方法权重。修正原入口从 step=0 更新导致的 off-by-one。

```bash
cd /home/jrw/code_vpwem_momart_bbox_memory
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 /home/jrw/anaconda3/envs/vpwem/bin/python \
  -m momart_boxed_no_history.train
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=5 /home/jrw/anaconda3/envs/vpwem/bin/python \
  -m momart_boxed_no_history.evaluate \
  --checkpoint /home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100/models/model_25000.pt \
  --train-config /home/jrw/vpwem_outputs/momart_boxed_no_history_bs256_150k/seed100/train_config.yaml \
  --dataset-path /home/jrw/vpwem_data/momart_zarr/table_cleanup_to_sink_expert.zarr \
  --env-name table_cleanup_to_sink --env-type expert \
  --work-dir /home/jrw/vpwem_outputs/momart_boxed_no_history_eval/25k/sink_expert \
  --result-path /home/jrw/vpwem_outputs/momart_boxed_no_history_eval/25k/sink_expert/result.npy
```

代码目录和输出独立，归档 No-History 与原来的 memory 设计合同未覆盖。本轮验证的是画框输入是否有帮助，并不能据此判断历史 memory 是否有效。

## 验证

```bash
/home/jrw/anaconda3/envs/vpwem/bin/python -m unittest \
  momart_boxed_no_history.test_painting momart_boxed_no_history.test_dataset -v
```

检查仅修改边框像素、训练和在线画框一致、两帧均替换、缺失不修改、十数据集原动作与其他模态一致、episode 最后一帧重复正确。短训练输出独立于正式实验。

2026-10-09：4 项测试通过，覆盖全部 10 数据集。实际 physical BS256 短训练完成恰好 20 次更新并保存 model_20.pt；原始数据只读。完整模拟器成功率评测尚未运行。
模型严格重新加载后，真实在线检测器 + 最近两帧画框 + EMA DP 的 50 步采样通过，输出有限值动作 [1,9,10]。
