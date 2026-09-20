import json
import os
import sys

report = {
    "python": sys.version,
    "prefix": sys.prefix,
    "executable": sys.executable,
    "imports": {},
}

for name in [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "h5py",
    "hydra",
    "omegaconf",
    "zarr",
    "gym",
    "gymnasium",
    "robomimic",
    "robosuite",
    "mujoco",
    "mani_skill",
    "mikasa_robo_suite",
    "sapien",
    "mplib",
    "pybullet",
    "gibson2",
    "cleandiffuser",
]:
    try:
        module = __import__(name)
        report["imports"][name] = {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
        }
    except Exception as exc:
        report["imports"][name] = {"ok": False, "error": repr(exc)}

try:
    import torch

    report["cuda"] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        value = (torch.ones(16, device="cuda") * 2).sum().item()
        report["cuda"]["tensor_test"] = value
except Exception as exc:
    report["cuda"] = {"available": False, "error": repr(exc)}

print(json.dumps(report, indent=2, ensure_ascii=False))

failed = [name for name, result in report["imports"].items() if not result["ok"]]
if failed or not report.get("cuda", {}).get("available"):
    raise SystemExit(1)
