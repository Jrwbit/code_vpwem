"""Create a conservative, non-destructive bbox cache from detector sidecars.

The source sidecars are kept intact.  This gate favors precision: ambiguous
multi-candidate detections and low-score boxes become missing observations;
the gate never fabricates a box or carries one across an episode boundary.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr


from .quality import VERSION, quality_gate



def build_validated_sidecar(source, destination, threshold=0.35):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("destination must not overwrite source")
    src = zarr.open_group(str(source), mode="r")
    required = ("boxes", "valid", "ambiguous", "scores")
    if any(key not in src for key in required):
        raise ValueError("source sidecar is missing a required array")
    accepted = quality_gate(src["boxes"], src["valid"], src["ambiguous"],
                            src["scores"], threshold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    dst = zarr.open_group(str(destination), mode="w")
    for key in src.array_keys():
        data = np.asarray(src[key])
        dst.create_dataset(key, data=data, chunks=src[key].chunks)
    dst["valid"][:] = accepted
    attrs = dict(src.attrs.asdict())
    attrs.update(annotation_standard=VERSION,
                 quality_threshold=float(threshold),
                 accepted_frames=int(accepted.sum()),
                 rejected_ambiguous=int((np.asarray(src["valid"]) & np.asarray(src["ambiguous"])).sum()),
                 rejected_low_score=int((np.asarray(src["valid"]) & ~np.asarray(src["ambiguous"])
                                         & (np.asarray(src["scores"]) < threshold)).sum()),
                 manual_review_required=True)
    dst.attrs.update(attrs)
    return dict(frames=len(accepted), accepted=int(accepted.sum()),
                acceptance_rate=float(accepted.mean()),
                rejected=int((~accepted).sum()), destination=str(destination))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    source_root, destination_root = map(Path, (args.source_root, args.destination_root))
    destination_root.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(source_root.glob("*.bbox.zarr")):
        destination = destination_root / source.name
        records.append(build_validated_sidecar(source, destination, args.threshold))
    manifest = dict(version=VERSION, threshold=args.threshold, sidecars=records)
    (destination_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
