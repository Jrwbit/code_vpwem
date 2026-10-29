"""Render ten aligned original/ROI pairs with explicit missing-detection labels."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from export_momart_roi_video_pair import build_roi_frames


def encoder(path, width):
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x432", "-framerate", "30", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-crf", "20",
        "-profile:v", "baseline", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)
    ], stdin=subprocess.PIPE)


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,width,height,r_frame_rate,duration", "-of", "json", str(path)
    ], text=True))["streams"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, help="ZIP containing ten NPZ sources and source_manifest.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 13)
    title_font = ImageFont.truetype(font_path, 18)
    report = {"speed": 3, "source_fps": 20, "output_fps": 30,
              "sampling": "every second source frame; rounding within one output frame",
              "roi_display": "gray labeled placeholder when missing; actual model receives zeros", "tasks": []}
    with zipfile.ZipFile(args.sources) as archive:
        source_manifest = json.loads(archive.read("source_manifest.json"))
        assert len(source_manifest["videos"]) == 10
        for entry in source_manifest["videos"]:
            task = entry["dataset"]
            with archive.open(task + ".npz") as src, np.load(src) as data:
                rgb, boxes, valid = data["rgb"], data["boxes"], data["valid"]
            assert len(rgb) == entry["frames"] and int(valid.sum()) == entry["valid_frames"]
            roi = build_roi_frames(rgb, boxes, valid)
            indices = list(range(0, len(rgb), 2))
            paths = {kind: args.output / f"{task}_{kind}_3x.mp4" for kind in ("original", "roi", "comparison")}
            processes = {kind: encoder(path, 960 if kind == "comparison" else 480) for kind, path in paths.items()}
            try:
                for index in indices:
                    original = Image.new("RGB", (480, 432), "#edf0f3")
                    crop = original.copy()
                    original.paste(Image.fromarray(rgb[index]).resize((480, 360), Image.Resampling.NEAREST), (0, 44))
                    od, rd = ImageDraw.Draw(original), ImageDraw.Draw(crop)
                    od.text((10, 4), "ORIGINAL | 3x", font=title_font, fill="#172b3a")
                    rd.text((10, 4), "ROI 64x64 | 3x", font=title_font, fill="#172b3a")
                    for draw in (od, rd):
                        draw.text((10, 27), task, font=font, fill="#172b3a")
                        draw.text((10, 410), f"Source frame {index+1}/{len(rgb)} | {entry['demo']}", font=font, fill="#172b3a")
                    if valid[index]:
                        crop.paste(Image.fromarray(roi[index]).resize((320, 320), Image.Resampling.NEAREST), (80, 70))
                        rd.text((10, 48), "VALID DETECTION", font=font, fill="#087443")
                    else:
                        rd.rectangle((80, 70, 399, 389), fill="#d9dfe5")
                        rd.text((112, 198), "NO VALID DETECTION", font=title_font, fill="#a12626")
                        rd.text((121, 231), "Model ROI input = zeros", font=font, fill="#283b49")
                    comparison = Image.new("RGB", (960, 432))
                    comparison.paste(original, (0, 0)); comparison.paste(crop, (480, 0))
                    for kind, frame in (("original", original), ("roi", crop), ("comparison", comparison)):
                        processes[kind].stdin.write(frame.tobytes())
            finally:
                for process in processes.values():
                    process.stdin.close()
                codes = [process.wait() for process in processes.values()]
            assert codes == [0, 0, 0], codes
            videos = {}
            for kind, path in paths.items():
                info = probe(path)
                assert int(info["nb_frames"]) == len(indices)
                assert info["r_frame_rate"] == "30/1"
                assert abs(float(info["duration"])-len(rgb)/60) <= 1/30
                videos[kind] = {"file": path.name, **info}
            report["tasks"].append({"task": task, "demo": entry["demo"], "source_frames": len(rgb),
                                    "valid_frames": int(valid.sum()), "missing_frames": int((~valid).sum()),
                                    "sampled_valid_frames": int(valid[indices].sum()), "videos": videos})
            (args.output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"Completed {task}: {int(valid.sum())}/{len(rgb)} valid source frames", flush=True)
    with zipfile.ZipFile(args.output / "ten_tasks_original_and_roi_3x.zip", "w", compression=zipfile.ZIP_STORED) as package:
        for row in report["tasks"]:
            for kind in ("original", "roi"):
                name = row["videos"][kind]["file"]
                package.write(args.output/name, name)
        package.write(args.output/"manifest.json", "manifest.json")


if __name__ == "__main__":
    main()
