#!/usr/bin/env python3
"""Instantiate the experiment modules and print their real PyTorch module trees."""

import argparse
import copy

import torch


SHAPE_META = {
    "action": {"shape": [10]},
    "obs": {
        "rgb": {"shape": [3, 90, 120], "type": "rgb"},
        "rgb_wrist": {"shape": [3, 90, 120], "type": "rgb"},
        "depth": {"shape": [1, 90, 120], "type": "depth"},
        "depth_wrist": {"shape": [1, 90, 120], "type": "depth"},
        "proprio": {"shape": [10], "type": "low_dim"},
        "scan": {"shape": [1, 100], "type": "scan"},
    },
}


def count(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def raw_condition():
    from cleandiffuser.nn_condition import MultiImageObsConditionMomart
    return MultiImageObsConditionMomart(
        shape_meta=copy.deepcopy(SHAPE_META), emb_dim=256,
        rgb_model_name="resnet18", resize_shape=None,
        crop_shape=(75, 100), random_crop=True, use_group_norm=True,
        use_seq=True, keep_horizon_dims=True)


def build(name):
    from cleandiffuser.nn_condition import (
        DirectSpatialHistoryConditionMomart,
        TaskConditionedDirectObsEncoder,
        TaskConditionedEmbeddingEncoder,
        TaskConditionedObsEncoder,
    )
    from cleandiffuser.nn_diffusion.chitransformer import ChiTransformer
    from cleandiffuser.nn_diffusion.chitransformerptp import ChiTransformerPTP
    from cleandiffuser.nn_diffusion.longmem_chitransformerptp import LongMemChiTransformerPTP

    if name in {"dp", "dpptp"}:
        condition = raw_condition()
        cls = ChiTransformer if name == "dp" else ChiTransformerPTP
        diffusion = cls(
            10, 256, 9, 2, d_model=256, nhead=4, num_layers=8,
            timestep_emb_type="positional")
        return {
            "entry": "pipelines/dp_momart_image.py" if name == "dp" else "pipelines/dpptp_momart_image.py",
            "protocol": "raw image; To=2; Ta=9; Tm=0; DDPM steps=50",
            "condition": condition,
            "memory": None,
            "diffusion": diffusion,
        }

    if name == "vpwem":
        from cleandiffuser.nn_memory.qformer_memory import QFormerMemory
        condition = TaskConditionedEmbeddingEncoder(5, emb_dim=256)
        memory = QFormerMemory(
            mem_compress_length=2, emb_dim=256, num_layers=2,
            cache_compress="fifo", cache_max_length=8,
            use_pos_emb="sin_pos", subsample_ratio=20)
        diffusion = LongMemChiTransformerPTP(
            10, 256, 9, 2, 2, d_model=256, nhead=4, num_layers=8,
            p_drop_emb=0.0, p_drop_attn=0.3,
            timestep_emb_type="positional")
        return {
            "entry": "pipelines/fair_history_momart_multitask_emb.py",
            "protocol": "dp_emb; To=2; Ta=9; Tm=2; QFormer=2 layers/2 queries; FIFO=8; DDPM steps=50",
            "condition": condition,
            "memory": memory,
            "diffusion": diffusion,
        }

    if name in {"no_history", "direct", "direct_token2"}:
        if name == "no_history":
            condition = TaskConditionedObsEncoder(raw_condition(), 5, emb_dim=256)
            tm = 0
        else:
            direct = DirectSpatialHistoryConditionMomart(
                shape_meta=copy.deepcopy(SHAPE_META), emb_dim=256,
                rgb_model_name="resnet18", resize_shape=None,
                crop_shape=(75, 100), random_crop=True, use_group_norm=True,
                use_seq=True, keep_horizon_dims=True, pool_size=4,
                history_frames=16,
                history_output_tokens=2 if name == "direct_token2" else 0,
                history_subsample_ratio=20)
            condition = TaskConditionedDirectObsEncoder(direct, 5, emb_dim=256)
            tm = condition.history_token_count
        diffusion = LongMemChiTransformerPTP(
            10, 256, 9, 2, tm, d_model=256, nhead=4, num_layers=8,
            p_drop_emb=0.0, p_drop_attn=0.3,
            timestep_emb_type="positional")
        return {
            "entry": "pipelines/fair_history_momart_multitask.py",
            "protocol": f"raw image; To=2; Ta=9; Tm={tm}; DDPM steps=50",
            "condition": condition,
            "memory": None,
            "diffusion": diffusion,
        }
    raise ValueError(name)


def emit(name):
    built = build(name)
    print("=" * 100)
    print(f"MODEL: {name}")
    print(f"EXPERIMENT ENTRY: {built['entry']}")
    print(f"PROTOCOL: {built['protocol']}")
    print("=" * 100)
    grand_total = grand_trainable = 0
    for component in ("condition", "memory", "diffusion"):
        module = built[component]
        print(f"\n[{component.upper()}]")
        if module is None:
            print("None")
            continue
        total, trainable = count(module)
        grand_total += total
        grand_trainable += trainable
        print(module)
        print(f"PARAMETERS: total={total:,} trainable={trainable:,}")
    diffusion = built["diffusion"]
    print("\n[CROSS_ATTENTION_MASK]")
    print(diffusion.memory_mask.detach().cpu())
    print(f"\nTOTAL PARAMETERS (component sum): {grand_total:,}")
    print(f"TRAINABLE PARAMETERS: {grand_trainable:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=[
        "dp", "dpptp", "no_history", "direct", "direct_token2", "vpwem"])
    args = parser.parse_args()
    for name in args.models:
        emit(name)


if __name__ == "__main__":
    main()
