# VPWEM：带工作记忆与情景记忆的非马尔可夫视觉运动策略
# wandb key：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
## 安装

#### 1. 创建并激活 conda 环境

```bash
$ conda create -n vpwem python==3.9
$ conda activate vpwem
```

#### 2. 安装 PyTorch

根据你的 CUDA 版本安装兼容的 `torch`。下面是 CUDA 11.8 的示例：

```bash
$ pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

#### 3. 安装 MuJoCo

使用 `apt` 安装 `mujoco-py` 相关依赖：

```bash
$ sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
```

也可以使用 `conda` 安装：

```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```

然后把下面这行加入 `~/.bashrc`，并安装 `patchelf`：

```bash
export CPATH=$CONDA_PREFIX/include
source ~/.bashrc
pip install patchelf
```

按照 https://github.com/openai/mujoco-py#install-mujoco 安装 MuJoCo：

- 下载 MuJoCo 2.1 二进制文件：[Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) 或 [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz)。
- 解压得到的 `mujoco210` 目录到 `~/.mujoco/mujoco210`。

在 `~/.bashrc` 中配置下面的环境变量：

```bash
export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

安装 `mujoco-py`：

```bash
source ~/.bashrc
pip install "Cython<3" # mujoco-py 与 cython 3 不兼容
pip install mujoco-py==2.1.2.14
```

#### 4. 安装 Robosuite 和 Robomimic

```bash
pip install robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
pip install robomimic==0.2.0 # 复现实验时 robomimic 版本很重要
```

#### 5. 安装 Gym

```bash
pip install setuptools==65.5.0 pip==21  # 新版本打包工具会导致 gym 0.21 安装失败
pip install wheel==0.38.0
pip install gym==0.21.0 gym-notices==0.0.8
```

#### 6. 安装 CleanDiffuser 和 VPWEM

如果需要运行 `pipelines` 并复现实验结果，还需要安装对应的强化学习仿真环境。

```bash
pip install -r requirements.txt
pip install -e .
```

## 使用说明

### 仓库结构

```
vpwem/
├── cleandiffuser/          # 核心库
│   ├── dataset/            # 数据集加载器，例如 Robomimic、MoMaRT、MIKASA
│   ├── diffusion/          # 扩散模型，例如 DDPM、DDIM、EDM
│   ├── env/                # 环境封装，例如 Robomimic、MIKASA
│   ├── nn_condition/       # 条件网络，主要是图像观测编码器
│   ├── nn_diffusion/       # 扩散模型骨干网络，例如 ChiTransformer
│   ├── nn_memory/          # 记忆模块，例如 Q-Former、KMeans、AdjSim
│   └── utils/              # 工具函数
├── configs/dp/             # Hydra YAML 配置文件
│   ├── robomimic_multi_modal/  # Robomimic 基准配置
│   ├── momart/                 # MoMaRT 基准配置
│   └── mikasa/                 # MIKASA 基准配置
├── pipelines/              # 训练和评估脚本
├── exp_scripts/            # 批量运行实验的 shell 脚本
└── requirements.txt
```

### 配置组织方式

配置文件按基准任务和方法组织：

| 配置目录 | 方法 | 脚本前缀 |
| --- | --- | --- |
| `chi_transformer/` | Diffusion Policy (DP) | `dp_*` |
| `chi_transformer_ptp/` | 带 Past Token Prediction 的 DP，即 DP-PTP | `dpptp_*` |
| `mail/` | 带 Mamba Policy 的 DP，即 MaIL | `mail_*` |
| `chi_transformer_ptp_longmem/` | **VPWEM，本仓库方法** | `longmemdpptp_*` |

`pipelines` 目录下的脚本遵循下面的命名约定：

- `*.py`：训练脚本。
- `*_emb.py`：使用短上下文策略预先计算好的观测 embedding，训练速度更快。
- `*_eval.py` / `*_emb_eval.py`：评估脚本。

### 预先计算观测 Embedding（可选）

为了加速训练，可以先用训练好的短上下文策略给数据集写入观测 embedding。这个流程参考了 [Long-context Diffusion Policy](https://github.com/long-context-dp/ldp)。

```bash
python pipelines/rewrite_with_embeddings_*.py
```

以 Robomimic `square/ph` 为例，建议先复制一份数据集，不要直接改原始文件：

```bash
mkdir -p /home/wjr/code/vpwem/data/robomimic/datasets/square/ph_emb

cp /home/wjr/code/vpwem/data/robomimic/datasets/square/ph/image_abs.hdf5 \
   /home/wjr/code/vpwem/data/robomimic/datasets/square/ph_emb/image_abs_emb.hdf5
```

然后使用短上下文 checkpoint 写入 `dp_emb`：

```bash
python pipelines/rewrite_with_embeddings_robomimic.py \
  --config-name=square_abs \
  dataset_path=/home/wjr/code/vpwem/data/robomimic/datasets/square/ph_emb/image_abs_emb.hdf5 \
  model_path=/home/wjr/code/vpwem/outputs/robomimic/dpptp/square_ph/obs_2_act_8/100/models/model_1000000.pt \
  save_path=/home/wjr/code/vpwem/outputs \
  batch_size=64 \
  device=cuda:0
```

写入完成后，可以用下面的命令检查数据集中是否已经出现 `dp_emb`：

```bash
python - <<'PY'
import h5py

path = "/home/wjr/code/vpwem/data/robomimic/datasets/square/ph_emb/image_abs_emb.hdf5"

with h5py.File(path, "r") as f:
    demo = sorted(f["data"].keys())[0]
    print("demo:", demo)
    print("obs keys:")
    for k in f["data"][demo]["obs"].keys():
        print(" ", k)
PY
```

### 训练

运行前，需要在对应的 YAML 配置文件中设置下面两个字段：

- `dataset_path`：数据集路径。
- `save_path`：模型 checkpoint 保存路径。

也可以通过命令行直接覆盖任意配置项：

```bash
python pipelines/longmemdpptp_robomimic_image_emb.py \
    --config-name=longsquare_abs_emb \
    seed=100 \
    mem_compress_length=2
```

如果已经给 Robomimic `square/ph` 数据集写入了 `dp_emb`，可以运行 VPWEM / long memory embedding 版本：

```bash
python pipelines/longmemdpptp_robomimic_image_emb.py \
  --config-name=square_abs_emb \
  dataset_path=/home/wjr/code/vpwem/data/robomimic/datasets/square/ph_emb/image_abs_emb.hdf5 \
  save_path=/home/wjr/code/vpwem/outputs \
  env_type=ph \
  +inference_action_steps=8 \
  device=cuda:0
```

如果只是想先确认训练管线能跑通，可以小步数测试：

```bash
python pipelines/longmemdpptp_robomimic_image_emb.py \
  --config-name=square_abs_emb \
  dataset_path=/home/wjr/code/vpwem/data/robomimic/datasets/square/ph_emb/image_abs_emb.hdf5 \
  save_path=/home/wjr/code/vpwem/outputs \
  env_type=ph \
  +inference_action_steps=8 \
  gradient_steps=10 \
  save_freq=5 \
  log_freq=1 \
  device=cuda:0
```

### 评估

评估脚本和训练脚本使用同一套配置。运行前确认 `work_dir` 指向已经保存模型 checkpoint 的目录。

```bash
# 在 Robomimic LongSquare 上评估 VPWEM
python pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py \
    --config-name=longsquare_abs \
    seed=100
```

注意：部分评估脚本里还有硬编码的 checkpoint 路径或 checkpoint 步数，例如 `PATH_TO_SHORT_CONTEXT_POLICY_CKPT_MODEL` 和 `ckpt = 1000000`。正式评估前需要把这些值改成真实路径，或者把它们改成 Hydra 参数后从命令行传入。

### 使用 Shell 脚本批量运行

`exp_scripts/` 里提供了一些示例 shell 脚本，可以跨多个随机种子批量运行训练和评估。

## 致谢

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [Long-context Diffusion Policy](https://github.com/long-context-dp/ldp)
- [MaIL](https://github.com/ALRhub/MaIL)
