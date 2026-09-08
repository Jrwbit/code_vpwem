#!/usr/bin/env python3
"""Download the robot-memory papers named in the 2026-08-24 research note.

The collection is intentionally explicit: each row was resolved to an arXiv,
CVF, or publisher record before being added.  Downloads are atomic and safe to
resume.  A JSON manifest and a human-readable README are regenerated after
every run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


USER_AGENT = "vpwem-robot-memory-literature/1.0 (academic paper downloader)"


@dataclass(frozen=True)
class Paper:
    number: int
    key: str
    title: str
    year: int
    record_url: str
    pdf_url: str | None
    identifier: str
    note: str = ""

    @property
    def filename(self) -> str:
        return f"{self.number:02d}_{self.key}.pdf"


def arxiv(
    number: int,
    key: str,
    title: str,
    year: int,
    arxiv_id: str,
    note: str = "",
) -> Paper:
    return Paper(
        number=number,
        key=key,
        title=title,
        year=year,
        record_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        identifier=f"arXiv:{arxiv_id}",
        note=note,
    )


PAPERS = [
    Paper(
        1,
        "memory_gated_diffusion_policy",
        "Memory-gated Diffusion Policy: Advancing Robotic Behaviour Learning with Memory-oriented Architectures",
        2025,
        "https://doi.org/10.1016/j.knosys.2025.113738",
        None,
        "DOI:10.1016/j.knosys.2025.113738",
        "Publisher and OpenAlex mark this article closed access; no public author manuscript was located.",
    ),
    arxiv(2, "vpwem", "VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory", 2026, "2603.04910"),
    arxiv(3, "dssp", "DSSP: Diffusion State Space Policy with Full-History Encoding", 2026, "2605.14598"),
    arxiv(4, "long_context_diffusion_policy", "Training and Evaluating Diffusion Policies with Long Context Lengths", 2026, "2606.16447"),
    arxiv(5, "aem", "Action-Effect Memory Pretraining for Robot Manipulation", 2026, "2606.12499", "The latest arXiv version is downloaded (the note calls this AEM v2)."),
    arxiv(6, "vq_memory", "Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks", 2026, "2603.09513", "Contains the RuleSafe benchmark."),
    arxiv(7, "pam", "Resolving State Ambiguity in Robot Manipulation via Adaptive Working Memory Recoding", 2025, "2512.24638", "Method acronym: PAM."),
    arxiv(8, "histrise", "History-Aware Visuomotor Policy Learning via Point Tracking", 2025, "2509.17141", "Method/project name: HistRISE."),
    arxiv(9, "memoryvla", "MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation", 2025, "2508.19236"),
    arxiv(10, "memoryvla_pp", "MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models", 2026, "2606.09827"),
    arxiv(11, "mem", "MEM: Multi-Scale Embodied Memory for Vision Language Action Models", 2026, "2603.03596"),
    arxiv(12, "memer", "MemER: Scaling Up Memory for Robot Control via Experience Retrieval", 2025, "2510.20328"),
    arxiv(13, "sam2act", "SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation", 2025, "2501.18564", "The paper includes SAM2Act+, MemoryBench, and their evaluations."),
    arxiv(14, "hamlet", "HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy", 2025, "2510.00695"),
    arxiv(15, "map_vla", "MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation", 2025, "2511.09516"),
    arxiv(16, "eventvla", "EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies", 2026, "2606.20092"),
    arxiv(17, "lamem_vla", "Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation", 2026, "2607.07608", "Method name: LaMem-VLA."),
    arxiv(18, "hime", "HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control", 2026, "2607.03449"),
    arxiv(19, "optimusvla", "Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation", 2026, "2602.20200", "Method name: OptimusVLA."),
    arxiv(20, "chameleon", "Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation", 2026, "2603.24576", "The same arXiv record was later retitled Control-Indexed Prospective Memory for Visuomotor Manipulation."),
    arxiv(21, "echovla", "EchoVLA: Robotic Vision-Language-Action Model with Synergistic Declarative Memory for Mobile Manipulation", 2025, "2511.18112"),
    arxiv(22, "helm", "HELM: Harness-Enhanced Long-horizon Memory for Vision-Language-Action Manipulation", 2026, "2604.18791"),
    arxiv(23, "cronusvla", "CronusVLA: Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling", 2025, "2506.19816"),
    arxiv(24, "hif_vla", "HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models", 2025, "2512.09928"),
    arxiv(25, "dysta", "Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement", 2026, "2602.03983", "The method was called SD-VLA in an early version and DySta in a later version; this is one paper, not two."),
    arxiv(26, "kc_vla", "Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining", 2026, "2603.01465", "Method name: Keyframe-Chaining VLA."),
    arxiv(27, "embodied_slotssm", "Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective", 2025, "2511.11478", "Introduces Embodied-SlotSSM and LIBERO-Mem."),
    arxiv(28, "cyclemanip", "CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding", 2025, "2512.01022"),
    arxiv(29, "rb_vla", "Recursive Belief Vision Language Model", 2026, "2602.20659", "Method name: RB-VLA."),
    arxiv(30, "mu_vla", "muVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models", 2026, "2606.12497"),
    arxiv(31, "stemvla", "StemVLA: An Open-Source Vision-Language-Action Model with Future 3D Spatial Geometry Knowledge and 4D Historical Representation", 2026, "2602.23721"),
    Paper(
        32,
        "mindexplore",
        "Towards Long-Horizon Vision-Language-Action System: Reasoning, Acting and Memory",
        2025,
        "https://openaccess.thecvf.com/content/ICCV2025/html/Li_Towards_Long-Horizon_Vision-Language-Action_System_Reasoning_Acting_and_Memory_ICCV_2025_paper.html",
        "https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Towards_Long-Horizon_Vision-Language-Action_System_Reasoning_Acting_and_Memory_ICCV_2025_paper.pdf",
        "ICCV:Li_2025_ICCV",
        "System name: MindExplore.",
    ),
    arxiv(33, "goal2skill", "Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection", 2026, "2604.13942"),
    arxiv(34, "soma", "SOMA: Strategic Orchestration and Memory-Augmented System for Vision-Language-Action Model Robustness via In-Context Adaptation", 2026, "2603.24060"),
    arxiv(35, "explicit_language_memory", "Explicit Language Memory for Long-Horizon Planning in Vision-Language-Action Models", 2026, "2608.04765"),
    arxiv(36, "baton", "Don't Drop the BATON: Long-Horizon Robot Manipulation via Agentic Subtask Exploration and Transition-aware Memory", 2026, "2608.16889"),
    arxiv(37, "robomme", "RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies", 2026, "2603.04639"),
    arxiv(38, "robomemarena", "RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark", 2026, "2605.10921"),
    arxiv(39, "tau0_vla", "tau0-VLA: A Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation", 2026, "2608.16885"),
    arxiv(40, "sparkvla", "SparkVLA: Stop-Aware Hierarchical VLA with Adaptive Action Chunking for Long-Horizon Manipulation", 2026, "2608.16172"),
    arxiv(41, "gated_memory_policy", "Gated Memory Policy: In-Context Memorization and Adaptation", 2026, "2604.18933", "Method acronym: GMP; introduces the MemMimic benchmark."),
]


def is_pdf(path: Path) -> bool:
    try:
        return path.stat().st_size > 50_000 and path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(path: Path) -> int | None:
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def request_bytes(url: str, timeout: int = 240) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_one(paper: Paper, pdf_dir: Path, retries: int) -> dict[str, Any]:
    destination = pdf_dir / paper.filename
    row: dict[str, Any] = {
        **asdict(paper),
        "filename": paper.filename,
        "path": str(destination),
    }
    if paper.pdf_url is None:
        return {
            **row,
            "status": "access_restricted",
            "error": "No legal public PDF source located",
        }
    if is_pdf(destination):
        pages = pdf_page_count(destination)
        return {
            **row,
            "status": "existing",
            "bytes": destination.stat().st_size,
            "pages": pages,
            "sha256": sha256(destination),
        }

    last_error: Exception | None = None
    candidate_urls = [paper.pdf_url]
    if "arxiv.org/pdf/" in paper.pdf_url:
        candidate_urls.append(paper.pdf_url.replace("https://arxiv.org/", "https://export.arxiv.org/"))
    for candidate_url in candidate_urls:
        for attempt in range(1, retries + 1):
            temporary = destination.with_suffix(".pdf.part")
            try:
                data = request_bytes(candidate_url)
                if len(data) <= 50_000 or not data.startswith(b"%PDF-"):
                    raise RuntimeError(f"response is not a valid PDF ({len(data)} bytes)")
                temporary.write_bytes(data)
                pages = pdf_page_count(temporary)
                if pages is None or pages < 1:
                    raise RuntimeError("pdfinfo could not validate the downloaded file")
                temporary.replace(destination)
                return {
                    **row,
                    "status": "downloaded",
                    "download_url": candidate_url,
                    "bytes": destination.stat().st_size,
                    "pages": pages,
                    "sha256": sha256(destination),
                }
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(min(2**attempt, 12))
    return {**row, "status": "failed", "error": str(last_error)}


def write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest = {
        "collection": "Robot manipulation memory papers named in the 2026-08-24 research note",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "paper_count": len(rows),
        "downloaded_or_existing": sum(row["status"] in {"downloaded", "existing"} for row in rows),
        "access_restricted": sum(row["status"] == "access_restricted" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "papers": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Robot Memory Paper Collection (2026-08-24)",
        "",
        "This directory contains the independent papers named in the supplied research note.",
        "Names that refer to a benchmark inside another paper are not duplicated: SAM2Act includes",
        "SAM2Act+/MemoryBench; Embodied-SlotSSM includes LIBERO-Mem; VQ-Memory includes RuleSafe.",
        "SD-VLA and DySta are versions/names of the same arXiv record.",
        "",
        f"- Total independent papers: {len(rows)}",
        f"- PDFs present: {manifest['downloaded_or_existing']}",
        f"- Access-restricted: {manifest['access_restricted']}",
        f"- Download failures: {manifest['failed']}",
        "",
        "| # | Paper | Identifier | PDF status | Notes |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        status = row["status"]
        if status in {"downloaded", "existing"}:
            status_text = f"[{status}](pdfs/{row['filename']})"
        else:
            status_text = status
        note = row.get("note", "")
        if row.get("error"):
            note = f"{note} {row['error']}".strip()
        title = row["title"].replace("|", "\\|")
        note = note.replace("|", "\\|")
        lines.append(
            f"| {row['number']} | [{title}]({row['record_url']}) | {row['identifier']} | {status_text} | {note} |"
        )
    lines.extend(
        [
            "",
            "`manifest.json` records byte size, page count, SHA-256 checksum, source URL, and status.",
            "Rerun the downloader to retry failures or refresh the reports without replacing valid PDFs.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("literature/vpwem/robot_memory_papers_2026-08-24"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output.resolve()
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Collection: {output_dir}", flush=True)
    print(f"Papers: {len(PAPERS)}", flush=True)

    rows_by_number: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_one, paper, pdf_dir, max(1, args.retries)): paper
            for paper in PAPERS
        }
        for future in concurrent.futures.as_completed(futures):
            paper = futures[future]
            try:
                row = future.result()
            except Exception as error:  # keep the collection resumable after an unexpected error
                row = {
                    **asdict(paper),
                    "filename": paper.filename,
                    "path": str(pdf_dir / paper.filename),
                    "status": "failed",
                    "error": repr(error),
                }
            rows_by_number[paper.number] = row
            print(f"[{row['status']:<17}] {paper.number:02d} {paper.key}", flush=True)

    rows = [rows_by_number[paper.number] for paper in PAPERS]
    write_reports(output_dir, rows)
    successful = sum(row["status"] in {"downloaded", "existing"} for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    restricted = sum(row["status"] == "access_restricted" for row in rows)
    print(f"Summary: present={successful} restricted={restricted} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
