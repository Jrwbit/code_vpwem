"""VPWEM 的训练编排、在线记忆更新与 DDPM 采样。

``LongMemDDPM`` 是项目的算法中枢，但它不单独定义网络结构。它把
``condition``（视觉/状态编码器）、``memory``（历史压缩器）和 ``diffusion``
（动作 Transformer）装进同一个 ``ModuleDict``，从动作去噪损失端到端优化，
并维护一份仅用于稳定推理的 EMA 副本。
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import amp

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.nn_diffusion.chitransformer import ChiTransformer
from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
try:
    from cleandiffuser.nn_diffusion.mail import MaIL
    from cleandiffuser.nn_diffusion.longmem_mail import LongMemMaIL
except ModuleNotFoundError:
    MaIL = None
    LongMemMaIL = None
from cleandiffuser.nn_memory.base_nn_memory import BaseNNMemory
from cleandiffuser.nn_memory.random_memory import RandomMemory
from cleandiffuser.nn_memory.kmeans_memory import KMeansMemory
from cleandiffuser.nn_memory.adjsim_memory import AdjSimMemory
from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
from cleandiffuser.utils import (
    at_least_ndim,
    cosine_beta_schedule,
    linear_beta_schedule,
    to_tensor)
from cleandiffuser.utils.mlp_correlation import MLP
from cleandiffuser.utils.mine_utils import Mine
from .basic import DiffusionModel
from copy import deepcopy
from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP
from cleandiffuser.nn_diffusion.direct_history_chitransformerptp import DirectHistoryChiTransformerPTP


def apply_memory_intervention(memory: Optional[torch.Tensor], mode: str = "none"):
    """Apply a causal intervention to compressed memory at inference time."""
    if memory is None or mode in (None, "none"):
        return memory
    if mode == "zero":
        return torch.zeros_like(memory)
    if mode == "shuffle_batch":
        if memory.shape[0] < 2:
            raise ValueError("shuffle_batch requires at least two samples")
        return memory.roll(shifts=1, dims=0)
    if mode == "reverse_tokens":
        return memory.flip(dims=(1,))
    raise ValueError(f"Unknown memory intervention: {mode}")


class LongMemDDPM(DiffusionModel):
    """把感知、情景记忆和动作 diffusion 串成一个可训练/可 rollout 的策略。

    训练数据流：
        ``history -> condition encoder -> memory compressor -> memory tokens``，
        ``current obs -> condition encoder -> observation tokens``，随后二者共同
        条件化带噪动作的预测。

    在线数据流：保留最近 ``obs_steps`` 个编码观测作为 working memory；更早
    的观测逐步交给 ``nn_memory.inference``。``reset_cache`` 与记忆模块的
    ``reset`` 应在 episode 边界调用。
    """

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,
            nn_memory: Optional[BaseNNMemory] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.999,
            optim_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,
            short_cond_dropout: float = 0.0,
            long_cond_dropout: float = 0.0,

            # ------------------- DPM Params ------------------- #
            predict_noise: bool = True,
            beta_schedule: str = "cosine",  # or cosine
            beta_schedule_params: Optional[dict] = None,
            device: Union[torch.device, str] = "cpu",

            # ----- action predictability ------ #
            action_range: Optional[torch.Tensor] = None,
            action_min: Optional[torch.Tensor] = None,

            # ----- others ------ #
            args_dict: Optional[dict] = None,
    ):
        # super().__init__(
        #     nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
        #     diffusion_steps, ema_rate, optim_params, device)

        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-5}

        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate

        # nn_condition is None means that the model is not conditioned on any input.
        self.use_condition_model =  nn_condition is not None
        self.nn_condition_drop = IdentityCondition(dropout=short_cond_dropout)
        self.nn_memory_drop = IdentityCondition(dropout=long_cond_dropout)

        # In the code implementation of Diffusion models, it is common to maintain an exponential
        # moving average (EMA) version of the model for inference, as it has been observed that
        # this approach can result in more stable generation outcomes.
        self.model = nn.ModuleDict({
            "diffusion": nn_diffusion.to(self.device),
            "condition": nn_condition.to(self.device) if nn_condition is not None else None,
            "memory": nn_memory.to(self.device) if nn_memory is not None else None,})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        
        self.model.train()
        self.model_ema.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)
        self.scaler = amp.GradScaler()

        self.classifier = classifier

        self.fix_mask = to_tensor(fix_mask, self.device)[None, ] if fix_mask is not None else 0.
        self.loss_weight = to_tensor(loss_weight, self.device)[None, ] if loss_weight is not None else 1.

        self.predict_noise = predict_noise

        if beta_schedule_params is None:
            beta_schedule_params = {}
        beta_schedule_params["T"] = self.diffusion_steps

        if beta_schedule == "linear":
            beta = linear_beta_schedule(**beta_schedule_params)
        elif beta_schedule == "cosine":
            beta = cosine_beta_schedule(**beta_schedule_params)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)
        self.x_max, self.x_min = x_max, x_min
        self.action_dim = (
            int(x_max.shape[-1]) if x_max is not None
            else int(nn_diffusion.act_emb.in_features))
        if isinstance(nn_diffusion, (
                ChiTransformerPTP, LongMemChiTransformerPTP,
                DirectHistoryChiTransformerPTP)):
            self.action_steps = nn_diffusion.T + 1 - nn_diffusion.To
            self.obs_steps = nn_diffusion.To
        elif isinstance(nn_diffusion, ChiTransformer):
            self.action_steps = nn_diffusion.T
            self.obs_steps = nn_diffusion.To
        elif LongMemMaIL is not None and isinstance(nn_diffusion, LongMemMaIL):
            self.action_steps = nn_diffusion.action_steps
            self.obs_steps = nn_diffusion.obs_steps
        elif MaIL is not None and isinstance(nn_diffusion, MaIL):
            self.action_steps = nn_diffusion.action_steps
            self.obs_steps = nn_diffusion.obs_steps
        
        self.inference_action_steps = int(getattr(
            args_dict, "inference_action_steps",
            getattr(self, "action_steps", 1)))
        # self.use_condition_cache = (self.obs_steps > self.action_steps)
        # if self.use_condition_cache:
        self.condition_cache_list = []
        self.action_cache_list = []
        self.spatial_frame_cache_list = []
        self.direct_selected_history = []
        self.direct_history_step = 0
        
        self.pred_reg_coeff = args_dict.pred_reg_coeff
        self.memory_aux_coeff = float(
            getattr(args_dict, "memory_aux_coeff", 0.0))
        self.counterfactual_memory_coeff = float(
            getattr(args_dict, "counterfactual_memory_coeff", 0.0))
        self.counterfactual_memory_margin = float(
            getattr(args_dict, "counterfactual_memory_margin", 0.01))
        self.counterfactual_memory_probability = float(
            getattr(args_dict, "counterfactual_memory_probability", 1.0))
        self.counterfactual_negative_mode = str(
            getattr(args_dict, "counterfactual_negative_mode", "random"))
        self.counterfactual_objective = str(
            getattr(args_dict, "counterfactual_objective", "ranking"))
        valid_counterfactual_modes = {
            "none", "random", "matched", "unmatched_branch"}
        if self.counterfactual_negative_mode not in valid_counterfactual_modes:
            raise ValueError(
                "counterfactual_negative_mode must be one of "
                f"{sorted(valid_counterfactual_modes)}, got "
                f"{self.counterfactual_negative_mode!r}")
        valid_counterfactual_objectives = {"ranking", "donor_action"}
        if self.counterfactual_objective not in valid_counterfactual_objectives:
            raise ValueError(
                "counterfactual_objective must be one of "
                f"{sorted(valid_counterfactual_objectives)}, got "
                f"{self.counterfactual_objective!r}")
        if self.pred_reg_coeff:
            self.action_pred_criterion = nn.MSELoss(reduction='mean')
            self.dataset_action_pred_ratio = np.load(args_dict.dataset_action_pred_ratio_path).item()
            self.dataset_action_pred_mlp = MLP(self.action_dim, 512, self.action_dim).to(device)
            dataset_action_pred_mlp_sd = torch.load(
                args_dict.dataset_action_pred_mlp_path, map_location=self.device, weights_only=False)
            self.dataset_action_pred_mlp.load_state_dict(dataset_action_pred_mlp_sd)
            self.dataset_action_pred_mlp.eval()
            for name, param in self.dataset_action_pred_mlp.state_dict().items():
                param.requires_grad = False
            self.action_range = action_range.to(device)
            self.action_min = action_min.to(device)

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps
        bar_alpha = at_least_ndim(self.bar_alpha[t], x0.dim())
        xt = x0 * bar_alpha.sqrt() + eps * (1 - bar_alpha).sqrt()
        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask
        return xt, t, eps
    
    def reverse_noise(self, xt, t, eps):
        bar_alpha = at_least_ndim(self.bar_alpha[t], xt.dim())
        x0 = (xt - eps * (1 - bar_alpha).sqrt()) / bar_alpha.sqrt()
        x0 = x0 * (1. - self.fix_mask) + xt * self.fix_mask
        return x0

    def loss(
            self, x0, condition=None, memory=None, ep_step=None,
            counterfactual_index=None, counterfactual_weight=None,
            memory_context=None,
            use_ema=False):
        """计算 VPWEM 的训练目标。

        ``x0`` 为 ``[B, Ta, action_dim]`` 的归一化动作；``condition`` 是最近
        ``To`` 帧；``memory`` 是当前窗口之前的可变长历史；``ep_step`` 提供
        历史帧在 episode 内的位置。基础目标是动作去噪 MSE，配置开启时再叠加
        memory auxiliary loss 与 counterfactual-memory loss。

        返回值中的 ``memory_tensor`` 是编码后、压缩前的历史，
        ``compressed_memory`` 是策略真正读取的 ``[B, Tm, D]`` token。
        """
        xt, t, eps = self.add_noise(x0)
        model = self.model_ema if use_ema else self.model
        memory_tensor = None
        raw_compressed_memory = None
        compressed_memory = None
        shared_long_keep_mask = None
        with amp.autocast('cuda', dtype=torch.float16):
            # MultiImageObsConditionRobomimic flattens values by assigning back
            # into its input mapping.  Keep caller-owned dictionaries intact so
            # repeated Monte-Carlo loss draws see the same tensor shapes.
            condition_input = (
                dict(condition) if isinstance(condition, dict) else condition)
            condition = self.nn_condition_drop(model["condition"](condition_input) if ((condition_input is not None) and self.use_condition_model) else condition_input)
            # 原始图像 pipeline 给出多模态 dict，需要先与当前观测共享编码器；
            # *_emb pipeline 已预计算 dp_emb，因而可直接进入记忆压缩器。
            if memory is not None:
                if isinstance(memory, dict):
                    # Multi-modal memory: process through nn_condition first
                    if self.use_condition_model:
                        if hasattr(model["condition"], "encode_spatial_history"):
                            memory_tensor = model["condition"].encode_spatial_history(
                                dict(memory))
                        else:
                            memory_tensor = model["condition"](
                                dict(memory))  # (B, T, D) when keep_horizon_dims=True
                    else:
                        raise ValueError("Memory is a dict but nn_condition is None. Cannot process multi-modal memory.")
                else:
                    # Single tensor memory (e.g., dp_emb_mem): use directly
                    memory_tensor = memory
                
                # 这是 VPWEM 相对普通 Diffusion Policy 的核心插入点：把可变长
                # T_history 压成固定 Tm，使下游 Transformer 的条件长度有界。
                if model["memory"] is not None:
                    raw_compressed_memory = (
                        model["memory"](
                            memory_tensor, ep_step, **memory_context)
                        if memory_context
                        else model["memory"](memory_tensor, ep_step))
                else:
                    raw_compressed_memory = memory_tensor
                # For a ranking pair, correct and donor histories must have
                # exactly the same long-condition availability. Applying the
                # legacy dropout before index_select gives every donor its own
                # mask and lets the hinge exploit "present vs absent" instead
                # of history semantics. Sample one mask per anchor and reuse it
                # after gathering the donor below.
                if self.nn_memory_drop.training:
                    shared_long_keep_mask = (
                        torch.rand(
                            raw_compressed_memory.shape[0],
                            device=raw_compressed_memory.device)
                        > self.nn_memory_drop.dropout
                    ).to(torch.float32)
                    compressed_memory = raw_compressed_memory * at_least_ndim(
                        shared_long_keep_mask, raw_compressed_memory.dim())
                else:
                    compressed_memory = self.nn_memory_drop(
                        raw_compressed_memory)
            else:
                compressed_memory = None
        # The condition encoders run under AMP, while the decoder forward below
        # intentionally remains in its native dtype. Legacy training dropout
        # happened to promote FP16 outputs through a FP32 mask; make that
        # boundary explicit so shared masks and offline eval behave identically.
        if isinstance(condition, torch.Tensor):
            condition = condition.to(dtype=xt.dtype)
        if compressed_memory is not None:
            compressed_memory = compressed_memory.to(dtype=xt.dtype)
        counterfactual_requested = (
            self.counterfactual_memory_coeff > 0
            and self.counterfactual_negative_mode != "none"
            and compressed_memory is not None
            and compressed_memory.shape[0] > 1
        )
        # The ranking comparison must not be driven by two independent decoder
        # dropout masks. Save the state immediately before the correct forward
        # and restore it before the wrong-history forward below.
        cpu_rng_state = None
        cuda_rng_state = None
        if counterfactual_requested:
            cpu_rng_state = torch.random.get_rng_state()
            if xt.is_cuda:
                cuda_rng_state = torch.cuda.get_rng_state(xt.device)

        if self.predict_noise:
            eps_hat = model["diffusion"](xt, t, condition, compressed_memory)
            loss = (eps_hat - eps) ** 2
            if self.pred_reg_coeff:
                x0_hat = self.reverse_noise(xt, t, eps_hat)
        else:
            x0_hat = model["diffusion"](xt, t, condition, compressed_memory)
            loss = (x0_hat - x0) ** 2

        weighted_loss = loss * self.loss_weight * (1 - self.fix_mask)
        correct_per_sample = weighted_loss.reshape(
            weighted_loss.shape[0], -1).mean(dim=1)
        action_loss = correct_per_sample.mean()

        if self.pred_reg_coeff:
            x0_hat_unnorm = (x0_hat + 1.) / 2.
            x0_hat_unnorm = x0_hat_unnorm * self.action_range + self.action_min
            x = x0_hat_unnorm[:, :-1, :]  # Shape: (N, S-1, dim)
            y = x0_hat_unnorm[:, 1:, :]   # Shape: (N, S-1, dim)

            # Reshape to combine batch and sequence dimensions
            x = x.reshape(-1, self.action_dim)  # Shape: ((N*(S-1)), dim)
            y = y.reshape(-1, self.action_dim)  # Shape: ((N*(S-1)), dim)
            y_hat = self.dataset_action_pred_mlp(x)
            action_pred_ratio = self.action_pred_criterion(y_hat, y)
            reg_loss = (action_pred_ratio - self.dataset_action_pred_ratio) ** 2
            total_loss = action_loss + self.pred_reg_coeff * reg_loss
        else:
            total_loss = action_loss

        memory_aux_loss = torch.zeros((), device=action_loss.device)
        if (
                self.memory_aux_coeff > 0
                and model["memory"] is not None
                and hasattr(model["memory"], "get_aux_loss")):
            memory_aux_loss = model["memory"].get_aux_loss()
            total_loss = total_loss + self.memory_aux_coeff * memory_aux_loss

        counterfactual_loss = torch.zeros((), device=action_loss.device)
        wrong_memory_loss = torch.zeros((), device=action_loss.device)
        counterfactual_gap = torch.zeros((), device=action_loss.device)
        counterfactual_ratio = torch.zeros((), device=action_loss.device)
        margin_violation_rate = torch.zeros((), device=action_loss.device)
        counterfactual_pairs = torch.zeros((), device=action_loss.device)
        use_counterfactual = False
        if counterfactual_requested:
            batch_size = compressed_memory.shape[0]
            if counterfactual_weight is None:
                pair_weight = torch.ones(
                    batch_size, device=action_loss.device,
                    dtype=correct_per_sample.dtype)
            else:
                pair_weight = torch.as_tensor(
                    counterfactual_weight, device=action_loss.device,
                    dtype=correct_per_sample.dtype).reshape(-1)
                if pair_weight.numel() != batch_size:
                    raise ValueError(
                        "counterfactual_weight must have one value per sample")
                pair_weight = pair_weight.clamp_min(0)
            probability_mask = (
                torch.rand(batch_size, device=action_loss.device)
                < self.counterfactual_memory_probability)
            active = (pair_weight > 0) & probability_mask
            use_counterfactual = bool(torch.any(active).item())

        if use_counterfactual:
            if self.counterfactual_negative_mode == "random":
                donor_index = torch.arange(
                    compressed_memory.shape[0],
                    device=compressed_memory.device).roll(shifts=1)
            else:
                if counterfactual_index is None:
                    raise ValueError(
                        f"{self.counterfactual_negative_mode} counterfactual "
                        "training requires explicit counterfactual_index")
                counterfactual_index = torch.as_tensor(
                    counterfactual_index, device=compressed_memory.device,
                    dtype=torch.long).reshape(-1)
                if counterfactual_index.numel() != compressed_memory.shape[0]:
                    raise ValueError(
                        "counterfactual_index must have one value per sample")
                if torch.any(counterfactual_index < 0) or torch.any(
                        counterfactual_index >= compressed_memory.shape[0]):
                    raise ValueError("counterfactual_index is out of batch range")
                donor_index = counterfactual_index
            wrong_memory = raw_compressed_memory.index_select(0, donor_index)
            if shared_long_keep_mask is not None:
                wrong_memory = wrong_memory * at_least_ndim(
                    shared_long_keep_mask, wrong_memory.dim())
            wrong_memory = wrong_memory.to(dtype=xt.dtype)

            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, xt.device)
            if self.counterfactual_objective == "donor_action":
                # True counterfactual supervision: keep the anchor's complete
                # current observation, substitute the donor history, and make
                # the denoising target the donor branch's expert action.  The
                # same t/eps are reused so differences cannot come from noise.
                donor_x0 = x0.index_select(0, donor_index)
                donor_xt, _, _ = self.add_noise(donor_x0, t=t, eps=eps)
                wrong_prediction = model["diffusion"](
                    donor_xt, t, condition, wrong_memory)
                wrong_target = eps if self.predict_noise else donor_x0
            else:
                wrong_prediction = model["diffusion"](
                    xt, t, condition, wrong_memory)
                wrong_target = eps if self.predict_noise else x0
            wrong_loss = (wrong_prediction - wrong_target) ** 2
            wrong_weighted_loss = (
                wrong_loss * self.loss_weight * (1 - self.fix_mask))
            wrong_per_sample = wrong_weighted_loss.reshape(
                wrong_weighted_loss.shape[0], -1).mean(dim=1)
            active_weight = pair_weight * active.to(pair_weight.dtype)
            normalizer = active_weight.sum().clamp_min(1.0)
            if self.counterfactual_objective == "donor_action":
                counterfactual_loss = (
                    wrong_per_sample * active_weight).sum() / normalizer
                hinge = torch.zeros_like(wrong_per_sample)
            else:
                hinge = torch.relu(
                    self.counterfactual_memory_margin
                    + correct_per_sample
                    - wrong_per_sample)
                counterfactual_loss = (
                    hinge * active_weight).sum() / normalizer
            wrong_memory_loss = (
                wrong_per_sample * active_weight).sum() / normalizer
            if self.counterfactual_objective == "ranking":
                gaps = wrong_per_sample - correct_per_sample
                counterfactual_gap = (
                    gaps * active_weight).sum() / normalizer
                ratios = gaps / correct_per_sample.detach().clamp_min(1e-8)
                counterfactual_ratio = (
                    ratios * active_weight).sum() / normalizer
                margin_violation_rate = (
                    (hinge > 0).to(active_weight.dtype) * active_weight
                ).sum() / normalizer
            counterfactual_pairs = active.to(active_weight.dtype).sum()
            total_loss = (
                total_loss
                + self.counterfactual_memory_coeff * counterfactual_loss
            )

        self._last_memory_losses = {
            "memory_aux_loss": memory_aux_loss.detach(),
            "counterfactual_memory_loss": counterfactual_loss.detach(),
            "counterfactual_donor_action_loss": (
                counterfactual_loss.detach()
                if self.counterfactual_objective == "donor_action"
                else torch.zeros_like(counterfactual_loss.detach())),
            "wrong_memory_action_loss": wrong_memory_loss.detach(),
            "counterfactual_gap": counterfactual_gap.detach(),
            "counterfactual_ratio": counterfactual_ratio.detach(),
            "counterfactual_margin_violation": margin_violation_rate.detach(),
            "counterfactual_pairs": counterfactual_pairs.detach(),
            "counterfactual_applied": torch.as_tensor(
                float(use_counterfactual), device=action_loss.device),
        }
        return total_loss, memory_tensor, compressed_memory, action_loss

   
    def update(
            self, x0, condition=None, memory=None, ep_step=None,
            counterfactual_index=None, counterfactual_weight=None,
            memory_context=None,
            update_ema=True, optimizer_step=True, loss_scale=1.0, **kwargs):
        # Step 1: Update the main model (diffusion + condition + memory) with MI loss
        total_loss, memory_tensor, compressed_memory, action_loss = self.loss(
            x0, condition, memory, ep_step,
            counterfactual_index=counterfactual_index,
            counterfactual_weight=counterfactual_weight,
            memory_context=memory_context)
        self.scaler.scale(total_loss * float(loss_scale)).backward()
        grad_norm = None
        if optimizer_step:
            if self.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if update_ema:
                self.ema_update()
        log = {
            "loss": total_loss.item(),
            "grad_norm": grad_norm,
            "action_loss": action_loss.item(),
            "optimizer_step": float(optimizer_step),
        }
        if hasattr(self, "_last_memory_losses"):
            log.update({
                key: value.item()
                for key, value in self._last_memory_losses.items()
            })

        return log

    def update_classifier(self, x0, condition):
        xt, t, eps = self.add_noise(x0)
        log = self.classifier.update(xt, t, condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def predict_function(
            self, x, t, bar_alpha,
            use_ema=False, requires_grad=False,
            # ----------------- CFG ----------------- #
            condition_vec_cfg=None,
            w_cfg: float = 0.0,
            # ----------------- CG ----------------- #
            condition_vec_cg=None,
            w_cg: float = 1.0,
            memory=None,
    ):
        b = x.shape[0]
        model = self.model_ema if use_ema else self.model

        # ----------------- CFG ----------------- #
        with torch.set_grad_enabled(requires_grad):
            # if w_cfg != 0.0 and w_cfg != 1.0:
            #     repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
            #     condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
            #     pred = model["diffusion"](
            #         x.repeat(*repeat_dim), t.repeat(2), condition_vec_cfg)
            #     pred = w_cfg * pred[:b] + (1. - w_cfg) * pred[b:]
            # elif w_cfg == 0.0:
            #     pred = model["diffusion"](x, t, None)
            # else:
            pred = model["diffusion"](x, t, condition_vec_cfg, memory)

        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x.clone(), t, condition_vec_cg)
            if self.predict_noise:
                pred = pred - w_cg * (1 - bar_alpha).sqrt() * grad
            else:
                pred = pred + w_cg * (1 - bar_alpha) / bar_alpha.sqrt() * grad
        else:
            log_p = None

        if self.predict_noise:
            if self.clip_pred:
                upper_bound = (x - bar_alpha.sqrt() * self.x_min) / (1 - bar_alpha).sqrt() \
                    if self.x_min is not None else None
                lower_bound = (x - bar_alpha.sqrt() * self.x_max) / (1 - bar_alpha).sqrt() \
                    if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            pred = pred * (1 - self.fix_mask)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)
            pred = pred * (1 - self.fix_mask) + x * self.fix_mask

        return pred, {"log_p": log_p}

    def reset_cache(self):
        self.condition_cache_list = []
        self.action_cache_list = []
        self.spatial_frame_cache_list = []
        self.direct_selected_history = []
        self.direct_history_step = 0

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            memory_cfg=None,
            ep_step_cfg=None,
            action_context_cfg=None,
            memory_intervention: str = "none",
            memory_index=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """用 DDPM 反向过程生成一个动作 chunk，并维护在线 working memory。

        有两种记忆输入方式：传入 ``memory_cfg`` 时一次性压缩显式历史，适合
        离线评估；否则函数复用跨调用的 ``condition_cache_list``，把滑出最近
        ``obs_steps`` 窗口的观测增量写入记忆模块，适合环境 rollout。

        ``memory_intervention`` / ``memory_index`` 只在压缩后替换或扰动记忆，
        用于检验动作是否真的因历史而改变，不属于标准采样必需步骤。
        """
        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions and memories
        direct_spatial = (
            self.use_condition_model
            and hasattr(model["condition"], "encode_spatial_frames")
            and hasattr(model["condition"], "pack_history")
            and getattr(model["condition"], "history_frames", None)
            is not None)
        if direct_spatial:
            # Match VPWEM's online evidence schedule exactly: frames only enter
            # history after leaving [t-1,t], and one is retained every r steps.
            with torch.set_grad_enabled(requires_grad):
                if memory_cfg is not None:
                    condition_vec_cfg = model["condition"](
                        dict(condition_cfg), mask_cfg)
                    compressed_memory = model["condition"].encode_spatial_history(
                        dict(memory_cfg))
                    self.condition_cache_list = [
                        condition_vec_cfg[:, i]
                        for i in range(condition_vec_cfg.shape[1])
                    ][-self.obs_steps:]
                    self.spatial_frame_cache_list = []
                    self.direct_selected_history = []
                    self.direct_history_step = 0
                else:
                    if not self.condition_cache_list:
                        new_input = dict(condition_cfg)
                    else:
                        new_input = {
                            key: (
                                value
                                if value.ndim <= 1 or key.startswith("_")
                                else value[:, -self.inference_action_steps:])
                            for key, value in condition_cfg.items()}
                    new_condition = model["condition"](
                        dict(new_input), mask_cfg)
                    new_spatial = model["condition"].encode_spatial_frames(
                        dict(new_input))
                    self.condition_cache_list.extend([
                        new_condition[:, i]
                        for i in range(new_condition.shape[1])])
                    self.spatial_frame_cache_list.extend([
                        new_spatial[:, i]
                        for i in range(new_spatial.shape[1])])

                    history_candidates = (
                        self.spatial_frame_cache_list[:-self.obs_steps])
                    ratio = int(getattr(
                        model["condition"], "history_subsample_ratio", 1))
                    for frame in history_candidates:
                        if self.direct_history_step % ratio == 0:
                            self.direct_selected_history.append(frame)
                        self.direct_history_step += 1
                    max_frames = model["condition"].history_frames
                    if len(self.direct_selected_history) > max_frames:
                        if max_frames == 1:
                            self.direct_selected_history = (
                                self.direct_selected_history[:1])
                        else:
                            self.direct_selected_history = (
                                self.direct_selected_history[:1]
                                + self.direct_selected_history[-(max_frames - 1):])

                    self.condition_cache_list = (
                        self.condition_cache_list[-self.obs_steps:])
                    self.spatial_frame_cache_list = (
                        self.spatial_frame_cache_list[-self.obs_steps:])
                    condition_vec_cfg = torch.stack(
                        self.condition_cache_list, dim=1)
                    if self.direct_selected_history:
                        selected = torch.stack(
                            self.direct_selected_history, dim=1)
                    else:
                        selected = new_spatial[:, :0]
                    compressed_memory = model["condition"].pack_history(
                        selected)
                condition_vec_cg = condition_cg
        elif memory_cfg is not None:
            # 路径 A：调用方显式提供完整历史（常用于离线诊断/成对干预）。
            with torch.set_grad_enabled(requires_grad):
                if condition_cfg is not None and self.use_condition_model:
                    condition_vec_cfg = model["condition"](condition_cfg, mask_cfg)
                else:
                    condition_vec_cfg = condition_cfg
                condition_vec_cg = condition_cg
                if condition_vec_cfg is not None and condition_vec_cfg.shape[1] > self.obs_steps:
                    condition_vec_cfg = condition_vec_cfg[:, -self.obs_steps:, :]
                if condition_vec_cfg is not None:
                    self.condition_cache_list = [
                        condition_vec_cfg[:, i, :] for i in range(condition_vec_cfg.shape[1])
                    ]
                compressed_memory = (
                    model["memory"](
                        memory_cfg, ep_step_cfg,
                        **({'action_history': action_context_cfg}
                           if getattr(
                               model["memory"],
                               'requires_action_history', False)
                           else {}))
                    if model["memory"] is not None and memory_cfg.shape[1] > 0
                    else None
                )
        # if self.use_condition_cache:
        elif len(self.condition_cache_list) == 0:
            # 路径 B1：episode 的第一次调用，建立最近 To 帧的 working memory。
            with torch.set_grad_enabled(requires_grad):
                if condition_cfg is not None and self.use_condition_model:
                    condition_vec_cfg = model["condition"](condition_cfg, mask_cfg)
                else:
                    condition_vec_cfg = condition_cfg
                condition_vec_cg = condition_cg
                if condition_vec_cfg is not None and condition_vec_cfg.shape[1] > self.obs_steps:
                    initial_history = condition_vec_cfg[:, :-self.obs_steps, :]
                    initial_kwargs = {}
                    if getattr(
                            model["memory"],
                            'requires_action_history', False):
                        action_dim = self.action_dim
                        initial_kwargs['action_history'] = (
                            initial_history.new_zeros(
                                initial_history.shape[0],
                                initial_history.shape[1], action_dim))
                    compressed_memory = model["memory"].inference(
                        initial_history, **initial_kwargs)
                else:
                    compressed_memory = None
                if condition_vec_cfg is not None:
                    for i in range(self.obs_steps, 0, -1):
                        self.condition_cache_list.append(condition_vec_cfg[:, -i, :])
                    action_dim = self.action_dim
                    self.action_cache_list = [
                        condition_vec_cfg.new_zeros(
                            condition_vec_cfg.shape[0], action_dim)
                        for _ in range(len(self.condition_cache_list))]
                    condition_vec_cfg = condition_vec_cfg[:, -self.obs_steps:, :]
        else:
            # 路径 B2：后续 rollout。加入新观测，把滑出 To 窗口的旧观测送入
            # nn_memory.inference，然后仅保留最新 To 帧供当前策略直接读取。
            with torch.set_grad_enabled(requires_grad):
                if not self.use_condition_model:
                    condition_vec_cfg = condition_cfg[:, -self.obs_steps:, :]
                    condition_vec_cg = condition_cg
                    compressed_memory = model["memory"].inference(condition_cfg[:, :-self.obs_steps, :]) if condition_cfg.shape[1] > self.obs_steps else None
                    self.condition_cache_list = [condition_vec_cfg[:, i, :] for i in range(condition_vec_cfg.shape[1])]
                else:
                    condition_cfg_new = dict()
                    for keys in condition_cfg.keys():
                        value = condition_cfg[keys]
                        condition_cfg_new[keys] = (
                            value
                            if value.ndim <= 1 or keys.startswith("_")
                            else value[:, -self.inference_action_steps:, ...])
                        mask_cfg_new = mask_cfg
                    condition_vec_cfg_new = model["condition"](condition_cfg_new, mask_cfg_new)
                    for i in range(self.inference_action_steps):
                        self.condition_cache_list.append(condition_vec_cfg_new[:, i, :])
                    if action_context_cfg is None:
                        action_dim = self.action_dim
                        action_context_cfg = condition_vec_cfg_new.new_zeros(
                            condition_vec_cfg_new.shape[0],
                            self.inference_action_steps, action_dim)
                    for i in range(self.inference_action_steps):
                        self.action_cache_list.append(
                            action_context_cfg[:, i, :])
                    condition_to_compress = self.condition_cache_list[:-self.obs_steps]
                    action_to_compress = self.action_cache_list[:-self.obs_steps]
                    self.condition_cache_list = self.condition_cache_list[-self.obs_steps:]
                    self.action_cache_list = self.action_cache_list[-self.obs_steps:]
                    condition_vec_cfg = torch.stack(self.condition_cache_list, dim=1)
                    condition_vec_cg = condition_cg
                    memory_kwargs = {}
                    if getattr(
                            model["memory"],
                            'requires_action_history', False):
                        memory_kwargs['action_history'] = torch.stack(
                            action_to_compress, dim=1)
                    compressed_memory = model["memory"].inference(
                        torch.stack(condition_to_compress, dim=1),
                        **memory_kwargs)
        # else:
        #     with torch.set_grad_enabled(requires_grad):
        #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
        #         condition_vec_cg = condition_cg
        
        if memory_index is not None:
            if compressed_memory is None:
                raise ValueError("memory_index requires non-empty compressed memory")
            memory_index = torch.as_tensor(
                memory_index, device=compressed_memory.device,
                dtype=torch.long).reshape(-1)
            if memory_index.numel() != compressed_memory.shape[0]:
                raise ValueError("memory_index must have one entry per sample")
            if torch.any(memory_index < 0) or torch.any(
                    memory_index >= compressed_memory.shape[0]):
                raise ValueError("memory_index is out of batch range")
            compressed_memory = compressed_memory.index_select(0, memory_index)
        compressed_memory = apply_memory_intervention(
            compressed_memory, memory_intervention)

        # 至此 condition_vec_cfg 是近期工作记忆，compressed_memory 是固定长度
        # 情景记忆；它们在每个 diffusion 反向步中保持不变。
        for t in range(self.diffusion_steps - 1, -1, -1):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_batch, bar_alpha,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg,
                memory=compressed_memory)

            # one step denoise
            if self.predict_noise:

                xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

            else:

                xt = 1 / (1 - bar_alpha) * (
                    alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                    beta * bar_alpha_prev.sqrt() * pred_theta)

            if t != 0:
                xt = xt + (beta * (1 - bar_alpha_prev) / (1 - bar_alpha)).sqrt() * torch.randn_like(xt)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log

    def sample_x(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            extra_sample_steps: int = 8,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # enter the sampling loop
        for t in range(self.diffusion_steps - 1, -1, -1):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_batch, bar_alpha,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)

            # one step denoise
            if self.predict_noise:

                xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

            else:

                xt = 1 / (1 - bar_alpha) * (
                    alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                    beta * bar_alpha_prev.sqrt() * pred_theta)

            if t != 0:
                xt = xt + (beta * (1 - bar_alpha_prev) / (1 - bar_alpha)).sqrt() * torch.randn_like(xt)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        if extra_sample_steps > 0:

            t_batch = torch.tensor(0, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[0]
            bar_alpha_prev = torch.tensor(1.0, device=self.device)
            alpha = self.alpha[0]
            beta = self.beta[0]

            for _ in range(extra_sample_steps):

                # predict eps_theta or x_theta with CG/CFG
                pred_theta, log = self.predict_function(
                    xt, t_batch, bar_alpha,
                    use_ema=use_ema,
                    requires_grad=requires_grad,
                    condition_vec_cfg=condition_vec_cfg,
                    condition_vec_cg=condition_vec_cg,
                    w_cfg=w_cfg, w_cg=w_cg)

                # one step denoise
                if self.predict_noise:

                    xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

                else:

                    xt = 1 / (1 - bar_alpha) * (
                            alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                            beta * bar_alpha_prev.sqrt() * pred_theta)

                # Fix the known portion
                xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log
