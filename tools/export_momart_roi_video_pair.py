"""Export aligned 3x original/ROI videos from RGB frames and cached xyxy boxes.

ROI construction matches momart_boxed_tokens.inputs.object_inputs: clip boxes,
floor/ceil edges, bilinear resize to 64x64 with align_corners=False; absent boxes
produce black frames. Display enlargement uses nearest-neighbor interpolation.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def write_video(path, frames, fps, scale):
    height, width = frames.shape[1:3]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
           "-an", "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    result = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                      "-show_entries", "stream=nb_frames,duration,width,height,r_frame_rate",
                                      "-of", "json", str(path)], text=True)
    stream = json.loads(result)["streams"][0]
    assert int(stream["nb_frames"]) == len(frames)
    assert abs(float(stream["duration"])-len(frames)/fps) < 1/fps
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-fps", type=float, default=20)
    parser.add_argument("--speed", type=float, default=3)
    args = parser.parse_args()
    assert args.source_fps > 0 and args.speed > 0
    with np.load(args.source) as data:
        rgb, boxes, valid = data["rgb"], data["boxes"], data["valid"]
    assert rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3
    assert boxes.shape == (len(rgb), 4) and valid.shape == (len(rgb),)
    roi = np.zeros((len(rgb), 64, 64, 3), dtype=np.uint8)
    height, width = rgb.shape[1:3]
    for i in np.flatnonzero(valid):
        assert np.isfinite(boxes[i]).all()
        box = np.clip(boxes[i], 0, 1)
        x1, y1 = np.floor(box[:2]*[width, height]).astype(int)
        x2, y2 = np.ceil(box[2:]*[width, height]).astype(int)
        assert x2 > x1 and y2 > y1
        crop = torch.from_numpy(rgb[i].copy()).permute(2, 0, 1).float()/255
        resized = F.interpolate(crop[None, :, y1:y2, x1:x2], size=(64, 64),
                                mode="bilinear", align_corners=False)
        roi[i] = (resized[0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    fps = args.source_fps*args.speed
    metadata = {"source": str(args.source), "frames": len(rgb), "valid_roi_frames": int(valid.sum()),
                "source_fps": args.source_fps, "speed": args.speed, "output_fps": fps,
                "missing_roi": "black; same zero input as policy", "videos": {}}
    for name, frames, scale in [("original_3x.mp4", rgb, 8), ("roi_3x.mp4", roi, 8)]:
        metadata["videos"][name] = write_video(args.output/name, frames, fps, scale)
    (args.output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
