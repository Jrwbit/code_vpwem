

# 这个文件是“训练入口/依赖装配层”：它不实现具体算法，
# 而是读取配置，创建数据、网络和优化器，然后启动训练循环。
# 阅读时可以把 args.xxx 都理解为 YAML 配置项。
#
# 本文形状符号：
#   B=batch size，To=近期观测数，T_history=压缩前历史数，
#   Tm=压缩后记忆 token 数，Ta=训练动作序列长度，D=token 维度。
# 常见 Python/PyTorch 写法：
#   getattr(args, "x", default) : 配置有 x 就取 x，否则取 default。
#   tensor.to(args.device)      : 将数据移到 CPU 或指定 GPU。
#   x[:, :To, ...]              : 所有 batch，只取前 To 个时间步。
#   **kwargs                     : 将字典展开为函数的命名参数。
import hydra  # 读取 YAML，并把命令行覆盖合并到 args。
import os  # 处理数据集和 checkpoint 路径。
import sys  # 当前未使用；历史入口代码遗留。
import warnings
# 禁止第三方库警告刷屏；调试环境问题时可临时注释掉。
warnings.filterwarnings('ignore')

import gym  # 当前入口未直接使用；是早期环境初始化遗留依赖。
import pathlib  # 将 work_dir 转成 Path 传给 Logger。
import time  # 统计训练的累计时间。
import collections  # defaultdict 用于累积两次日志之间的指标。
import numpy as np  # 兼容 NumPy action，并计算指标平均值。
import torch
import torch.nn as nn  # 当前未使用；保留自历史版本。
from utils import (
    set_seed, parse_cfg, Logger, auto_resume_training, prune_checkpoints)
from cleandiffuser.dataset.multitask import BalancedTaskBatchMux
# parse_cfg 在当前 Hydra 入口中未使用；其余函数分别负责随机种子、
# 日志/checkpoint、断点续训和删除过旧 checkpoint。
# CosineAnnealingLR 让学习率在训练过程中按余弦曲线逐渐降低。
from torch.optim.lr_scheduler import CosineAnnealingLR

# from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
# 把 MIKASA episode 切成“当前窗口 + 早期历史 + 动作标签”。
from cleandiffuser.dataset.mikasa_dataset_longmem import LongMemMikasaDataset
# loop_dataloader 无限轮询 DataLoader；collate_fn 补齐变长历史。
from cleandiffuser.dataset.dataset_utils import loop_dataloader, dynamic_batch_collate_fn
# NCM 实验用：把“配对样本的数据集索引”换成“batch 内位置”。
from cleandiffuser.dataset.ncm_pairing import partner_positions
# 打印可训练参数数量，便于核对模型规模。
from cleandiffuser.utils import report_parameters
# 下面四个 robomimic 工具在当前训练入口中未使用，属于历史遗留 import。
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# 默认读取 shell_game_touch.yaml。例如命令行传 lr=1e-5 可覆盖 YAML。
# Hydra 将合并后的 OmegaConf 对象作为 args 传入 pipeline。
@hydra.main(config_path="../configs/dp/mikasa/chi_transformer_ptp_longmem", config_name="shell_game_touch")
def pipeline(args):
    # 固定 Python/NumPy/PyTorch 随机数，让重复实验尽可能一致。
    set_seed(args.seed)
    # Logger 会在 work_dir 下保存配置、指标和模型。
    logger = Logger(pathlib.Path(args.work_dir), args)

    # PTP 的时间对齐约束。例如 To=2、执行动作数=8，Ta=horizon=9。
    # 之所以减 1，是因为第一帧观测与第一个动作共享起始时刻。
    assert args.horizon == args.obs_steps + args.action_steps - 1

    # 这是旧版 robomimic 模态注册逻辑，当前已不执行。
    # modality_mapping = collections.defaultdict(list)
    # for key, attr in args.shape_meta['obs'].items():
    #     modality_mapping[attr.get('type', 'low_dim')].append(key)
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
    # ==================== 1. 创建数据集 ====================
    # 这一块的任务不是训练网络，而是把一条完整 episode 切成训练样本。
    #
    # 一个 batch 里有两条时间线：
    #   obs[k]     : [B, To, ...]       当前时刻附近的 To 帧，作为短期条件；
    #   obs[k_mem] : [B, T_history, ...]当前窗口之前的历史，长度可变。
    # 其中 B=batch size，To=obs_steps。T_history 会由 collate_fn 在 batch 内补齐。
    # 注意：数据集只负责“选哪些历史帧”，不负责把它们压成 memory token。
    # expanduser 把 "~/data" 这种路径展开成绝对的用户目录路径。
    task_specs = getattr(args, "task_datasets", None)
    if task_specs:
        task_paths = {
            str(spec.name): os.path.expanduser(str(spec.path))
            for spec in task_specs
        }
    else:
        task_paths = {
            str(args.env_name): os.path.expanduser(str(args.dataset_path))
        }

    if len(task_paths) > 1 and getattr(args, "ncm_pair_manifest_path", None):
        raise ValueError("NCM pair manifests are not supported in multitask mode")

    def make_dataset(dataset_path):
        return LongMemMikasaDataset(
        dataset_dir=dataset_path,
        # 声明 action 维度以及每个 obs key 的形状和类型（rgb/low_dim）。
        shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps,  # 每个样本保留的近期观测帧数 To
        horizon=args.horizon,        # 目标动作序列长度 Ta
        # episode 开头/末尾凑不齐窗口时，复制边界帧做 padding。
        pad_before=args.obs_steps-1,
        pad_after=args.action_steps-1,
        # memory_type/subsample_ratio 决定从更早历史中如何取帧。
        dataset_memory_type=args.dataset_memory_type,
        subsample_ratio=args.dataset_memory_subsample_ratio,
        # 原始历史最多保留多少帧；这不是最终 memory token 数 Tm。
        maximum_memory_length=int(getattr(
            args, "dataset_maximum_memory_length", 16)),
        # 若为 True，历史过长需截断时仍强制保留 episode 第一帧。
        preserve_episode_anchor=bool(getattr(
            args, "dataset_preserve_episode_anchor", False)),
        # AEM-style 记忆除观测外还需历史动作；其他方法不额外读取。
        include_action_memory=(
            str(args.mem_compress_method) == "aem_style"),
        # 以下三项只有 NCM（negative/counterfactual memory）配对实验才会用。
        ncm_pair_manifest_path=getattr(args, "ncm_pair_manifest_path", None),
        ncm_pairing_seed=int(getattr(args, "ncm_pairing_seed", args.seed)),
        ncm_pair_batch_fraction=float(getattr(
            args, "ncm_pair_batch_fraction", 1.0)),
        # ShellGame 专用的数据转换选项：可在指定时刻插入若干保持帧。
        shellgame_hold_step=getattr(args, "shellgame_hold_step", None),
        shellgame_hold_steps=int(getattr(args, "shellgame_hold_steps", 0)),
        # 把原始 MIKASA 数据转成 ReplayBuffer 时的并行工作数和在途任务上限。
        conversion_n_workers=int(getattr(args, "conversion_n_workers", 4)),
        conversion_max_inflight_tasks=int(getattr(
            args, "conversion_max_inflight_tasks", 20)),
        # Dataset 的 normalizer 创建在此设备上。
        device=args.device,
        )

    datasets = {name: make_dataset(path) for name, path in task_paths.items()}
    dataset = next(iter(datasets.values()))
    # 可选：从另一份数据集重新计算并借用归一化统计量。
    # 常用于与旧 checkpoint 保持相同的 action/low_dim 尺度。
    normalizer_dataset_path = getattr(
        args, "normalizer_dataset_path", None)
    if normalizer_dataset_path:
        # 这个辅助 Dataset 只为获取 normalizer，load_images=False 避免白读图像。
        normalizer_dataset = LongMemMikasaDataset(
            dataset_dir=os.path.expanduser(str(normalizer_dataset_path)),
            shape_meta=args.shape_meta,
            n_obs_steps=args.obs_steps,
            horizon=args.horizon,
            pad_before=args.obs_steps - 1,
            pad_after=args.action_steps - 1,
            dataset_memory_type=args.dataset_memory_type,
            subsample_ratio=args.dataset_memory_subsample_ratio,
            maximum_memory_length=int(getattr(
                args, "dataset_maximum_memory_length", 16)),
            conversion_n_workers=int(getattr(
                args, "conversion_n_workers", 4)),
            conversion_max_inflight_tasks=int(getattr(
                args, "conversion_max_inflight_tasks", 20)),
            load_images=False,
            device=args.device,
        )
        # 覆盖主数据集自己的统计量，后续归一化都使用这一份。
        dataset.normalizer = normalizer_dataset.normalizer
        print(
            f"Using frozen-checkpoint normalizer from "
            f"{normalizer_dataset_path}")
    # __str__ 会打印 keys、总帧数和 episode 数，用来检查数据是否加载正确。
    for task_name, task_dataset in datasets.items():
        print(f"Training dataset [{task_name}]: {task_dataset}")
    
    # 变长历史无法直接 stack。dynamic_batch_collate_fn 会先把它们
    # 补到当前 batch 的最大 T_history；长度接近的样本放在一起可减少 padding。
    # NCM 实验还需保证预定义的成对样本处于同一 batch，所以优先用专用 sampler。
    # 配置没写 num_workers 时默认用 8 个子进程并行读数据。
    num_workers = int(getattr(args, "num_workers", 8))
    dataloaders = {}
    for task_name, task_dataset in datasets.items():
        ncm_batch_sampler = task_dataset.get_ncm_batch_sampler()
        if ncm_batch_sampler is not None:
            manifest_batch_size = int(
                task_dataset.ncm_manifest['batches'].shape[1])
            if manifest_batch_size != int(args.batch_size):
                raise ValueError(
                    f"NCM manifest batch size {manifest_batch_size} does not "
                    f"match configured batch_size {args.batch_size}")
            dataloaders[task_name] = torch.utils.data.DataLoader(
                task_dataset,
                batch_sampler=ncm_batch_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                collate_fn=dynamic_batch_collate_fn,
            )
            continue
        # 普通实验可按历史长度分组；如果 Dataset 没有专用 sampler 就随机打乱。
        train_sampler = task_dataset.get_dataset_sampler()
        dataloaders[task_name] = torch.utils.data.DataLoader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            # 最后不满 batch_size 的小 batch 丢掉，保持每步 batch 形状一致。
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=dynamic_batch_collate_fn,
        )
    batch_mux = BalancedTaskBatchMux(dataloaders, seed=args.seed)
        
    # ==================== 2. 组装感知和动作网络 ====================
    # nn_condition 是“感知器”：把 RGB/低维状态等多模态观测统一编码成 256 维 token。
    #   近期观测 [B, To, ...]        -> [B, To, 256]
    #   早期历史 [B, T_history, ...] -> [B, T_history, 256]
    # nn_diffusion 是“动作去噪器”：根据扩散时刻、近期 token 和记忆 token，
    # 对带噪动作 [B, Ta, action_dim] 预测噪声（或等价的去噪目标）。
    # 这里只构建两个网络，它们稍后会和 nn_memory 一起交给 LongMemDDPM。
    # 当前这个 pipeline 只支持 chi_transformer 骨干，分支保留了未来扩展点。
    if args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsConditionRobomimic
        from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
        
        nn_condition = MultiImageObsConditionRobomimic(
            # shape_meta 让编码器知道 rgb/rgb_wrist/joints 分别如何处理。
            shape_meta=args.shape_meta,
            emb_dim=256,  # 多个模态拼接/投影后，每帧输出 256 维。
            rgb_model_name=args.rgb_model,  # 默认 resnet18，用于提取每帧图像特征。
            resize_shape=args.resize_shape,  # 可选的统一缩放；None 表示不缩放。
            crop_shape=args.crop_shape,  # 送入 ResNet 前的裁剪大小。
            random_crop=args.random_crop,  # 训练时随机裁剪，相当于图像数据增强。
            # 将 ResNet 的 BatchNorm 换为 GroupNorm，小 batch 训练更稳定。
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,  # 输入包含时间维 [B, T, ...]。
            # 保留 T，输出 [B,T,256]；False 会把时间维展平。
            keep_horizon_dims=True,
        ).to(args.device)  # 模型参数也必须搬到与输入 tensor 相同的设备。
        nn_diffusion = LongMemChiTransformerPTP(
            act_dim=args.action_dim,  # 每个动作的维度，ShellGame 配置中为 8。
            obs_dim=256,  # 必须与 nn_condition 的 emb_dim 一致。
            Ta=args.horizon,  # 训练时带噪动作序列长度。
            To=args.obs_steps,  # 不压缩的近期观测 token 数。
            Tm=args.mem_compress_length,  # 压缩后的长期记忆 token 数。
            d_model=256,  # Transformer 内部隐藏维度。
            nhead=4,  # 多头注意力的 head 数。
            num_layers=args.diffusion_num_layers,  # Transformer decoder 层数。
            p_drop_emb=args.p_drop_emb,  # token embedding 的 dropout 概率。
            p_drop_attn=args.p_drop_attn,  # attention/FFN 中的 dropout 概率。
            # context=直接读全部记忆；gated=学习读取强度；
            # control_indexed=用当前控制状态检索相关记忆。
            memory_read_mode=str(getattr(
                args, "memory_read_mode", "context")),
            # 只在 gated 模式生效，控制记忆门的初始偏置。
            memory_gate_bias=float(getattr(
                args, "memory_gate_bias", 0.0)),
            # 把整数 diffusion step t 编码成向量。
            timestep_emb_type="positional",
        ).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    # ==================== 3. 选择记忆压缩方法 ====================
    # 所有方法都遵守同一接口：
    #   [B, T_history, 256] -> [B, Tm, 256]，Tm=mem_compress_length。
    # 因此只需改 YAML 中的 mem_compress_method，后面的扩散策略不用改。
    #
    # random/adjsim/kmeans : 非学习型基线，分别随机、相邻相似度、聚类选择历史；
    # qformer              : 用可学习 query 从历史中读取固定数量的 token；
    # per                  : 按事件重要性写入、合并和读取记忆；
    # *_style              : 从其他论文迁移的记忆策略。
    from cleandiffuser.nn_memory.random_memory import RandomMemory
    from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
    from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
    from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
    from cleandiffuser.nn_memory.per_memory import PERMemory
    if args.mem_compress_method == "random":
        # 从原始历史 token 中随机保留 Tm 个，作为最简单的对照组。
        nn_memory = RandomMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "kmeans":
        # 将历史特征聚成 Tm 类，用聚类中心概括整段历史。
        nn_memory = KMeansMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "adjsim":
        # 根据相邻帧特征的相似性去除冗余历史。
        nn_memory = AdjSimMemory(mem_compress_length=args.mem_compress_length).to(args.device)
    elif args.mem_compress_method == "qformer":
        # VPWEM 主方法：用 Tm 个可学习 query 通过 cross-attention 汇总历史。
        nn_memory = QFormerMemory(
            mem_compress_length=args.mem_compress_length,  # query/输出 token 数 Tm。
            emb_dim=args.qformer_emb_dim,  # 输入、query 和输出特征维度。
            num_layers=args.qformer_num_layers,  # Q-Former attention block 数。
            cache_compress=args.qformer_cache_compress,  # 推理 cache 满时的压缩/淘汰方式。
            cache_max_length=args.qformer_cache_max_length,  # 在线原始特征 cache 容量。
            use_pos_emb=args.memory_pos_emb,  # 历史时间位置的编码方式。
            # 把 ep_step 换算成与原 episode 时间尺度对应的位置。
            subsample_ratio=args.dataset_memory_subsample_ratio,
        ).to(args.device)
    elif args.mem_compress_method == "per":
        # PER 是事件驱动记忆：先按变化强度写入 bank，再读取成固定 token。
        nn_memory = PERMemory(
            mem_compress_length=args.mem_compress_length,  # 给策略的主记忆 token 数。
            evidence_length=int(getattr(args, "per_evidence_length", 0)),  # 额外证据 token 预算。
            emb_dim=args.qformer_emb_dim,
            num_layers=args.qformer_num_layers,
            num_heads=int(getattr(args, "qformer_num_heads", 8)),
            bank_size=args.qformer_cache_max_length,  # 事件记忆 bank 的最大容量。
            event_score_mode=str(getattr(args, "per_event_score", "delta")),  # 事件重要性评分方式。
            event_threshold=float(getattr(args, "per_event_threshold", 0.05)),  # 超过阈值才触发写入。
            max_write_interval=int(getattr(args, "per_max_write_interval", 20)),  # 最久多少步必须写一次。
            merge_similarity=float(getattr(args, "per_merge_similarity", 0.98)),  # 高于此相似度的事件可合并。
            score_decay=float(getattr(args, "per_score_decay", 0.995)),  # 旧事件重要性的衰减率。
            pos_emb_scale=float(getattr(args, "per_pos_emb_scale", 1.0)),  # 位置编码强度。
            subsample_ratio=args.dataset_memory_subsample_ratio,
        ).to(args.device)
    elif args.mem_compress_method in {
            "aem_style", "dssp_style", "vq_style", "cycle_progress"}:
        from cleandiffuser.nn_memory.paper_transfer_factory import (
            build_transfer_memory)
        # 这几种共用一个 factory，factory 再按方法名构建对应实现。
        nn_memory = build_transfer_memory(args).to(args.device)
    else:
        raise ValueError(f"Invalid memory method {args.mem_compress_method}")
    
    # 分别打印三个子模块的参数量，便于确认选中了预期的网络和层数。
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Memory Model =======================")
    report_parameters(nn_memory)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    # action 已经被 MinMaxNormalizer 归一化到 [-1, 1]。
    # x_min/x_max 告诉 diffusion sampler 最终动作序列的合法范围和形状。
    # 首维为 1，运算时会自动 broadcast 到实际 batch size。
    x_max = torch.ones(
        (1, args.horizon, args.action_dim), device=args.device) * +1.0
    x_min = torch.ones(
        (1, args.horizon, args.action_dim), device=args.device) * -1.0
    # 保留原始 action 的取值跨度和最小值，必要时可把 [-1,1] 还原回环境尺度。
    action_range = dataset.normalizer['action'].range.to(args.device)
    action_min = dataset.normalizer['action'].min.to(args.device)
    # Long-memory 入口目前只支持 DDPM；其他 diffusion 类型会明确报错。
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.longmem_ddpm import LongMemDDPM
        # agent 是训练外壳，它统一持有三个网络、AdamW、AMP scaler 和 EMA 副本。
        # agent.model     : 正在反向传播更新的参数。
        # agent.model_ema : 参数指数移动平均，通常用于更稳定的评估/推理。
        agent = LongMemDDPM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            nn_memory=nn_memory,
            device=args.device,
            # DDPM 前向加噪/反向去噪的离散时间步数。
            diffusion_steps=args.sample_steps,
            x_max=x_max,
            x_min=x_min,
            args_dict=args,  # 传入辅助 loss、NCM 等扩展配置。
            action_range=action_range,
            action_min=action_min,
            # 训练时随机丢弃近期/长期条件，提高条件鲁棒性。
            short_cond_dropout=args.short_cond_dropout,
            long_cond_dropout=args.long_cond_dropout,
            ema_rate=float(getattr(args, "ema_rate", 0.999)),
            optim_params={
                "lr": args.lr,
                "betas": (
                    float(getattr(args, "adam_beta1", 0.9)),
                    float(getattr(args, "adam_beta2", 0.999)),
                ),
            },
        )
    else:
        raise NotImplementedError(f"Diffusion method {args.diffusion} not supported for LongMem")
    
    # init_model_path 表示“只借用初始权重”：加载 model/EMA，但不恢复 optimizer、
    # scheduler 和当前 step，因此仍被视为一次新训练。
    init_model_path = getattr(args, "init_model_path", None)
    if init_model_path:
        # weights_only=False 表示 checkpoint 可包含普通 Python 对象，所以只应加载可信文件。
        init_checkpoint = torch.load(
            os.path.expanduser(str(init_model_path)),
            map_location=args.device, weights_only=False)
        # strict=True 要求 checkpoint 的每个参数键都与当前网络完全对上。
        init_strict = bool(getattr(args, "init_model_strict", True))
        # 分别加载训练模型和 EMA 模型；返回值会列出缺失/多余参数键。
        model_msg = agent.model.load_state_dict(
            init_checkpoint["model"], strict=init_strict)
        ema_msg = agent.model_ema.load_state_dict(
            init_checkpoint["model_ema"], strict=init_strict)
        print(f"Initialization model loaded with message: {model_msg}")
        print(f"Initialization EMA loaded with message: {ema_msg}")

    # 在 gradient_steps 个更新内将学习率按余弦轨迹从 lr 平滑降到接近 0。
    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=args.gradient_steps, last_epoch=-1)
    # 这与 init_model_path 不同：它搜索 work_dir 中最新的完整 model+scheduler，
    # 恢复训练状态并返回下一个 step；如果没找到则返回 0。
    n_gradient_step = auto_resume_training(
        agent, lr_scheduler, args.work_dir, args.gradient_steps)
    
    if args.mode == "train":
        # ==================== 4. 训练循环 ====================
        # loop_dataloader 会在一轮数据用完后继续下一轮，训练由 gradient_steps 终止。
        # 每一步的数据流：
        #   batch['obs'] -> condition（近期）+ memory（早期）
        #   batch['action'] -> naction（归一化的监督目标）
        #   agent.update(...) -> 加噪、编码、记忆压缩、去噪 loss、反传和 EMA
        # 例如 train_metric_lists['loss'] 是一个 list，累积 log_freq 步后再取平均。
        train_metric_lists = collections.defaultdict(list)
        start_time = time.time()
        # 除了固定 save_freq，也可在训练早期指定若干额外保存点。
        # getattr(..., []) 表示旧 YAML 没这个字段时使用空列表，保持兼容。
        early_save_steps = {
            int(step) for step in getattr(
                args, "checkpoint_save_early_steps", [])
        }
        # loop_dataloader 是无限迭代器；真正的退出条件在循环末尾。
        while True:
            task_name, batch = batch_mux.next(n_gradient_step)
            task_dataset = datasets[task_name]
            # 把 batch 拆成近期观测 condition 与早期历史 memory。
            # 原始图像版本中二者都是多模态 dict；LongMemDDPM.loss 会让它们
            # 共享同一个视觉编码器，再仅对 memory 分支做定长压缩。
            # nobs 是字典，典型 keys 包括：
            # rgb/rgb_wrist/joints（近期）、对应 *_mem（历史）、ep_step、actual_num_frames。
            nobs = batch['obs']
            # 两个字典最终的 key 名相同，但时间范围不同。
            # condition['rgb'] 是近期图像；memory['rgb'] 是早期历史图像。
            condition = {}
            memory = {}
            # 遍历所有观测模态，按 key 后缀将它们分流。
            for k in nobs.keys():
                if k.endswith('_mem'):
                    # image_mem -> image：去掉后缀后才能复用 image 的 normalizer。
                    base_key = k[:-4]  # Remove '_mem' suffix
                    con = nobs[k].to(args.device)  # [B, T_history, ...]
                    con = task_dataset.normalizer['obs'][base_key].normalize(con)
                    memory[base_key] = con
                elif k != 'ep_step' and k != 'actual_num_frames':
                    # ep_step 是历史位置，actual_num_frames 是真实长度，都不是观测模态。
                    con = nobs[k][:, :args.obs_steps, :].to(args.device)  # [B, To, ...]
                    con = task_dataset.normalizer['obs'][k].normalize(con)
                    condition[k] = con
            # DataLoader 通常已经把 action 组成 Tensor；NumPy 分支是兼容旧 collate 行为。
            # 原始形状是 [B, Ta, action_dim]。变量名 naction 表示 normalized action。
            naction = batch['action']
            if isinstance(naction, np.ndarray):
                naction = torch.from_numpy(naction).to(args.device)
            else:
                naction = naction.to(args.device)
            # 将环境动作归一化到约 [-1,1]，与 x_min/x_max 及 diffusion 训练尺度一致。
            naction = task_dataset.normalizer['action'].normalize(naction)
            
            # mem_steps 保留历史帧在 episode 中的真实时间位置，
            # 让记忆模块能区分“很早以前”和“刚刚发生”的内容。
            mem_steps = nobs['ep_step'].int().to(args.device)

            # 以下是可选实验输入。None 表示普通 VPWEM 训练不启用它们。
            counterfactual_index = None  # 每个样本应替换成哪个 donor history。
            counterfactual_weight = None  # 哪些配对有效，以及对应 loss 权重。
            memory_context = None  # 需要额外上下文的 memory method 使用。
            if 'memory_action' in batch:
                # 目前主要是 AEM-style 记忆需要与历史观测对齐的历史动作。
                memory_action = batch['memory_action'].to(args.device)
                memory_action = task_dataset.normalizer['action'].normalize(
                    memory_action)
                memory_context = {'action_history': memory_action}
            # none: 不做反事实记忆训练。
            # matched/unmatched_branch: 用配对样本的历史构造“错误记忆”。
            negative_mode = str(getattr(
                args, "counterfactual_negative_mode", "none"))
            if 'ncm' in batch:
                # ncm 保存 dataset index、分支标记和 donor 索引等配对元数据。
                ncm = batch['ncm']
                counterfactual_weight = ncm['alias_weight'].to(args.device)
                if negative_mode in {'matched', 'unmatched_branch'}:
                    partner_key = (
                        'matched_partner' if negative_mode == 'matched'
                        else 'unmatched_partner')
                    # manifest 中的 partner 是全数据集索引；agent 需要的是 0..B-1 的 batch 内位置。
                    counterfactual_index = partner_positions(
                        ncm['sample_index'].to(args.device),
                        ncm[partner_key].to(args.device),
                        valid=counterfactual_weight > 0)

            # 这是整个入口中最值得跟进去的一行。update 内部大致执行：
            #   1) 给 naction 加噪；
            #   2) 用 nn_condition 编码 condition 和 memory；
            #   3) 用 nn_memory 把 T_history 压成 Tm；
            #   4) 用 nn_diffusion 预测噪声并计算 loss；
            #   5) backward + optimizer.step + EMA 更新。
            # 下一步阅读应跳到 cleandiffuser/diffusion/longmem_ddpm.py 的 loss/update。
            update_log = agent.update(
                naction, condition, memory, mem_steps,
                counterfactual_index=counterfactual_index,
                counterfactual_weight=counterfactual_weight,
                memory_context=memory_context)
            # agent.update 已完成一次 optimizer.step，所以现在将学习率调度器推进一步。
            lr_scheduler.step()
            # update_log 至少含 loss/action_loss，也可能含梯度范数、memory/NCM loss。
            # 先累积原始数值，到 log_freq 时再统一求均值，减少单 batch 波动。
            for key, value in update_log.items():
                if value is not None:
                    train_metric_lists[key].append(float(value))

            # 每 log_freq 个梯度步记录一次。step=0 也会记录第一个 batch。
            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,  # 当前全局优化步。
                    'total_time': time.time() - start_time,  # 本次进程启动后的秒数。
                    'task_index': list(datasets).index(task_name),
                }
                # 字典推导式例如把 loss:[1.2,1.0] 变成 avg_loss:1.1。
                metrics.update({
                    f'avg_{key}': float(np.mean(values))
                    for key, values in train_metric_lists.items()
                    if values
                })
                # category='train' 将这些值归入训练指标。
                logger.log(metrics, category='train')
                # 写完后清空，下一次均值只覆盖新的时间窗口。
                train_metric_lists.clear()
            
            # step 0 不保存；命中额外 early step 或周期 save_freq 时保存。
            should_save = n_gradient_step > 0 and (
                n_gradient_step in early_save_steps
                or n_gradient_step % args.save_freq == 0)
            if should_save:
                # save_agent 保存训练模型、EMA 模型和 optimizer 状态。
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                # scheduler 单独保存，供 auto_resume_training 恢复正确学习率。
                torch.save(
                    lr_scheduler.state_dict(),
                    args.work_dir
                    + f'/models/scheduler_{str(n_gradient_step)}.pt')
                # 只保留最新的若干对 model+scheduler，避免 checkpoint 占满磁盘。
                prune_checkpoints(
                    args.work_dir, keep=int(args.get("keep_checkpoints", 3)))
            
            # 这个计数器按“已完成的 optimizer update”递增。
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # 达到设定训练步数后离开无限 dataloader 循环。
                # logger.finish(agent) 已被注释，因此这里不会额外保存“最终模型”。
                break
        
    else:
        # 评估有独立 *_eval.py 入口，本文件只接受 mode=train。
        raise ValueError("Illegal mode")


# 只有用 `python pipelines/longmemdpptp_mikasa_image.py` 直接运行时才启动。
# 如果该文件被 import，pipeline 不会自动执行。
if __name__ == "__main__":
    pipeline()
