# MoMaRT 带框 No-History BS64 1M 启动记录

2026-10-29，用户指定 GPU1、batch64、1M 步，超参沿用此前 1M No-History。已在 BHR（bitbot421）启动从零训练，PID 2690584；首步 loss=1.1116749048233032，lr=9.999999999975327e-05，physical/effective batch=64，accumulation=1。nvidia-smi 确认进程在 GPU1，占用约3742 MiB。训练仍在运行，尚无最终checkpoint或成功率。

## 配置与差异

基准为服务器 `/home/jrw/vpwem_outputs/fair_history_momart_multitask/no_history/seed100/.hydra/config.yaml`，完整副本及新配置、启动器保存在 `experiments/momart_boxed_nohistory_bs64_1m_20261029/`。

已用 OmegaConf 解析并逐键对照：全部训练超参保持原值。seed100；Adam lr=1e-4、betas=(0.9,0.999)；EMA=.999；cosine T_max=1000000；obs_steps=2、action_steps=8、horizon=9；DDPM50步；ResNet18、75×100随机裁剪；8层Transformer；short/long_cond_dropout=.3、attention dropout=.3。5任务10数据配置，长期token=0，无坐标或ROI token。

差异为增加 bbox_root；HDF5 改用既有 Zarr 缓存读取；独立输出目录；keep_checkpoints 从3改为10，以满足既有带框训练器保留所有里程碑的要求。save_freq仍100000，log_freq仍1000。使用既有带框训练器按实际完成更新计数，精确完成1M次更新；旧训练器从step0开始且循环条件为<=，存在一次更新的计数差异。随机读取路径变化，不承诺与旧实验逐batch/随机轨迹相同。

画框沿用150k带框实验：`/home/jrw/vpwem_data/momart_bbox_gdino_red_v1`，Grounding DINO tiny、RGB-only、prompt `a red bowl.`，仅最近两帧主RGB画1像素红框；无框保留原图。运行产物 bbox_input.json 已回读核对。

## 运行与复现

- 独立代码快照：`/home/jrw/code_vpwem_boxed_bs64_1m_20261029`，复制自既有 `code_vpwem_boxed_continue_20261027`，不影响已有训练/评测目录。
- 启动：`bash /home/jrw/code_vpwem_boxed_bs64_1m_20261029/experiments/momart_boxed_nohistory_bs64_1m_20261029/run.sh`，nohup后台运行，flock防止重复启动。
- 输出：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs64_1m_20261029/seed100/`。
- 日志：`/home/jrw/vpwem_outputs/momart_boxed_no_history_bs64_1m_20261029/train.log`。
- 检查：bash -n通过；10数据集加载完成；首步更新成功；实际配置差异核验通过。未停止卡1原有644 MiB进程。

代码快照关键文件SHA256：

- momart_boxed_no_history/train.py: e4bc3e6f29f50ddc9fde246e8b4b15b9511c940c3b16b22c7e3c601472bcdaa2
- momart_boxed_no_history/dataset.py: bdb466b938d478875cb4478f64c4f3bc7f01020b6288598d8ed0c139a0f25f3e
- momart_boxed_no_history/painting.py: 15a20a5b45e5e3b5e92fdccc1bc6ba2ccd0de8a58c4857f63b1ec555ae70a5a1
- momart_no_history_baseline/model.py: e94ec3e68885491c65cb85f2d46ab6d8ff29e5bf4ab74ba66e4302466d866aaf

飞书已发布并回读核验关键配置、PID及代码哈希（本记录无新增图像）：https://feishu.cn/docx/SlC8dHdpcolDPixTeyucMmeWnug 。

## 2026-10-30 08:57进度与预计完成时间

物理GPU1训练主进程2690584仍在运行。2026-10-29 15:58:11启动，已运行约17小时。最新持久化日志为450000/1000000步（45%），写入时间10月30日08:55:21；训练计时60973.7秒。已保存100k、200k、300k、400k四个模型。

最近10k/50k/100k步分别约0.13099/0.13159/0.13191秒每步。按剩余550k步推算，还需约20.0–20.2小时，预计 **10月31日05:00左右（北京时间）完成1M步训练**。这是速度外推，若资源竞争或训练速度变化，时间随之调整；不包含训练结束后的评测耗时。目前尚无该1M新模型的最终成功率。

证据：远端metrics.jsonl、模型文件列表和主进程运行时间；本次读取摘要保存在`artifacts/momart_status_20261030/verified.json`。
