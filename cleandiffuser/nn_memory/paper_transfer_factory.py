"""Factory for controlled paper-mechanism transfer experiments."""

from cleandiffuser.nn_memory.paper_transfer_memory import (
    AEMStyleMemory,
    CycleProgressMemory,
    DSSPStyleMemory,
    VQStyleMemory,
)


TRANSFER_MEMORY_METHODS = {
    "aem_style", "dssp_style", "vq_style", "cycle_progress"}


def build_transfer_memory(args):
    method = str(args.mem_compress_method)
    common = dict(
        mem_compress_length=int(args.mem_compress_length),
        emb_dim=int(args.qformer_emb_dim),
        num_layers=int(getattr(args, "transfer_num_layers", 1)),
        subsample_ratio=int(args.dataset_memory_subsample_ratio),
    )
    if method == "aem_style":
        return AEMStyleMemory(
            **common, action_dim=int(args.action_dim))
    if method == "dssp_style":
        return DSSPStyleMemory(**common)
    if method == "vq_style":
        return VQStyleMemory(
            **common,
            codebook_size=int(getattr(args, "vq_codebook_size", 64)),
            commitment_cost=float(getattr(
                args, "vq_commitment_cost", 0.25)),
        )
    if method == "cycle_progress":
        return CycleProgressMemory(**common)
    raise ValueError(f"Unknown transfer memory method {method!r}")
