#!/usr/bin/env python3
"""Enrich a coarse yearly VPWEM index with compact readable summaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DETAIL_API = "https://connectedresearchers.top/api/be/api/guide/paper"
VENUE_ORDER = ("ICLR", "ICML", "CVPR", "AAAI", "WACV", "AAMAS", "ACL", "EACL", "WSDM")
SUMMARY_OVERRIDES = {
    "iclr2026-54U3XHf7qq-981996a4fc": "MemoryVLA以工作记忆和长期感知-认知记忆库保存时序上下文，并用记忆条件扩散专家生成动作。",
    "iclr2026-1dH4ARGdwD-0aa94ae7c2": "MemER从长时机器人经验中检索任务相关关键帧，再由层级策略利用这些证据完成分钟级任务。",
    "iclr2026-IvUM6UwYCJ-8bad520056": "SeqWM用顺序世界模型刻画多机器人之间的协作过程，为跨主体时序建模提供参考。",
    "iclr2026-OeDwYtp8n1-64271e301d": "该方法通过协同轨迹控制生成机器人操作视频，重点处理多主体与多物体之间的运动一致性。",
    "iclr2026-aemqAxScl9-d56aaf448a": "SARM以阶段感知奖励模型处理长时程操作中的演示质量差异，并改善多阶段任务学习。",
    "iclr2026-LQD1MrnbxH-1f89c32997": "该方法在测试时动态混合多个世界模型，使具身智能体适应持续变化的环境动力学。",
    "iclr2026-y5CaJb17Fn-6ac90d96f8": "villa-X联合视觉、语言与潜在动作表示，增强VLA模型的动作规划和生成能力。",
    "icml2026-66214": "该方法为VLA维护视野外物体的空间记忆，使机器人在目标暂时不可见时仍能继续操作。",
    "aaai2026-article-37577": "TCoT通过轨迹思维链和失败恢复机制，提高VLA在长时程机器人操作中的纠错能力。",
    "acl2026-7cc20c97f79e": "该方法利用冲突检测规则改善具身智能体向量记忆的数据质量与任务规划可靠性。",
    "acl2026-c3258eb5eaab": "该工作为推理密集型检索引入情景记忆，以缓解仅依赖当前查询造成的信息遗忘。",
    "acl2026-192bf264ea8c": "GMSA通过分组合并与层级语义对齐压缩长上下文，属于可迁移的上下文压缩方法。",
    "dblp:conf/eacl/YeHCZ26": "H-Mem以混合多维记忆管理机制组织长上下文对话智能体的不同类型历史信息。",
    "dblp:conf/eacl/QingDMHV26": "该工作研究多跳长上下文推理中的记忆粒度选择，以及压缩粒度对证据保留的影响。",
}


def request_detail(paper_id: str, retries: int = 4) -> dict[str, Any]:
    url = f"{DETAIL_API}/{urllib.parse.quote(paper_id, safe='')}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 VPWEM-index/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == retries:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def first_sentence(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "暂无站内摘要，后续按需精读原文。"
    match = re.search(r"(?<=[。！？.!?])\s", text)
    sentence = text[: match.end()].strip() if match else text
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


def compact_text(text: str, limit: int = 500) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value


def choose_summary(paper_id: str, payload: dict[str, Any]) -> tuple[str, str]:
    if paper_id in SUMMARY_OVERRIDES:
        return SUMMARY_OVERRIDES[paper_id], "人工校正"
    guide = payload.get("guide") or {}
    analysis = guide.get("guide_analysis") or {}
    candidates = (
        analysis.get("one_liner_zh"),
        guide.get("guide_tldr_zh"),
        analysis.get("problem_zh"),
        guide.get("abstract_zh"),
    )
    for candidate in candidates:
        if candidate:
            return first_sentence(str(candidate)), "中文导读"
    paper = payload.get("paper") or {}
    return first_sentence(paper.get("abstract", "")), "英文摘要"


def vpwem_relation(title: str, priority: str) -> str:
    value = title.lower()
    if priority == "P0":
        if "benchmark" in value:
            return "直接评测依据：可用于补充记忆任务与指标。"
        if "memory trap" in value or "memorization" in value:
            return "关键反例：用于检验策略是否只是复现动作或轨迹。"
        return "直接相关：需比较记忆内容、更新方式、容量和单回合边界。"
    if "world model" in value:
        return "替代路线：比较预测未来与压缩过去哪种信息更有效。"
    if "action chunk" in value or "diffusion policy" in value or "flow policy" in value:
        return "策略基线：可用于控制动作生成器和时序建模的影响。"
    if "token compression" in value or "context compression" in value:
        return "压缩启发：可迁移到历史 token 选择、压缩率和效率实验。"
    if "streaming" in value:
        return "流式启发：关注在线更新以及成本是否随回合长度保持有界。"
    if "episodic" in value or "working memory" in value:
        return "记忆表示启发：可对照情景记忆与工作记忆的职责划分。"
    return "邻近工作：用于补充基线、Related Work 或实验边界。"


def enrich_one(record: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = request_detail(record["paper_id"])
        summary, summary_source = choose_summary(record["paper_id"], payload)
        paper = payload.get("paper") or {}
        guide = payload.get("guide") or {}
        analysis = guide.get("guide_analysis") or {}
        official = record.get("paper_url") or paper.get("forum_url") or paper.get("pdf_url") or ""
        code = record.get("github_url") or paper.get("github_url") or ""
        guide_text = compact_text(
            guide.get("abstract_zh") or analysis.get("abstract_zh") or summary
        )
        method_points = [compact_text(str(value), 180) for value in (analysis.get("method_zh") or [])[:2]]
        result_points = [compact_text(str(value), 180) for value in (analysis.get("result_zh") or [])[:2]]
        cover_image_url = guide.get("cover_image_url") or ""
        status = "ok"
    except Exception as exc:  # Keep the index complete even if one endpoint is temporarily unavailable.
        summary = SUMMARY_OVERRIDES.get(
            record["paper_id"], "详情接口暂时不可用，保留导读与原文链接供后续阅读。"
        )
        summary_source = "人工校正" if record["paper_id"] in SUMMARY_OVERRIDES else "抓取失败"
        official = record.get("paper_url", "")
        code = record.get("github_url", "")
        guide_text = summary
        method_points = []
        result_points = []
        cover_image_url = ""
        status = f"error:{type(exc).__name__}"
    return {
        **record,
        "guide_url": f"https://connectedresearchers.top/guide/paper/{urllib.parse.quote(record['paper_id'], safe='')}",
        "summary": summary,
        "summary_source": summary_source,
        "vpwem_relation": vpwem_relation(record["title"], record["priority"]),
        "paper_url": official,
        "github_url": code,
        "guide_text": guide_text,
        "method_points": method_points,
        "result_points": result_points,
        "cover_image_url": cover_image_url,
        "detail_status": status,
    }


def download_cover(record: dict[str, Any], covers_dir: Path) -> dict[str, Any]:
    url = record.get("cover_image_url", "")
    if not url:
        return record
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["paper_id"])
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 VPWEM-index/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "").lower()
        if content.startswith(b"\x89PNG") or "png" in content_type:
            suffix = ".png"
        elif content.startswith(b"\xff\xd8") or "jpeg" in content_type:
            suffix = ".jpg"
        else:
            suffix = ".webp"
        path = covers_dir / f"{safe_id}{suffix}"
        path.write_bytes(content)
        record["cover_image_path"] = str(path.resolve())
        record["cover_status"] = "ok"
    except Exception as exc:
        record["cover_image_path"] = ""
        record["cover_status"] = f"error:{type(exc).__name__}"
    return record


def write_outputs(output_dir: Path, source: dict[str, Any], records: list[dict[str, Any]]) -> None:
    year = int(source["year"])
    chinese_count = sum(item["summary_source"] == "中文导读" for item in records)
    summary_note = (
        "摘要优先采用 Connected Researchers 中文导读"
        if chinese_count
        else "站内未提供中文导读，本版采用论文英文摘要"
    )
    order = {venue: index for index, venue in enumerate(VENUE_ORDER)}
    records.sort(key=lambda item: (order.get(item["venue"], 99), item["priority"], -item["score"], item["title"]))
    output = {**source, "papers": records, "candidate_count": len(records)}
    (output_dir / f"vpwem_{year}_summaries.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = (
        "venue", "priority", "title", "relation", "summary", "vpwem_relation",
        "guide_url", "paper_url", "github_url", "summary_source", "detail_status",
    )
    with (output_dir / f"vpwem_{year}_summaries.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        f"# {year} 顶会与 VPWEM 相关论文：快速阅读版",
        "",
        f"更新时间：{dt.date.today().isoformat()}",
        "",
        f"共 {len(records)} 篇。{summary_note}；本版用于快速定位，关键结论仍需回到原文核验。",
        "",
        "阅读顺序：先看 P0，再看 P1；P2 仅在设计压缩或流式记忆实验时查阅。",
        "",
    ]
    for venue in VENUE_ORDER:
        selected = [item for item in records if item["venue"] == venue]
        lines.extend([f"## {venue} {year}", "", f"本轮收录 {len(selected)} 篇。", ""])
        if not selected:
            lines.extend(["本轮未发现明确相关论文。", ""])
            continue
        for priority in ("P0", "P1", "P2"):
            group = [item for item in selected if item["priority"] == priority]
            if not group:
                continue
            lines.extend([f"### {priority}", ""])
            for item in group:
                links = [f"[导读]({item['guide_url']})"]
                if item["paper_url"]:
                    links.append(f"[原文]({item['paper_url']})")
                if item["github_url"]:
                    links.append(f"[代码]({item['github_url']})")
                lines.extend(
                    [
                        f"**{item['title']}**",
                        "",
                        f"- 一句话：{item['summary']}",
                        f"- 对 VPWEM：{item['vpwem_relation']}",
                        f"- 链接：{' · '.join(links)}",
                        "",
                    ]
                )
    (output_dir / f"vpwem_{year}_summaries_feishu.md").write_text("\n".join(lines), encoding="utf-8")

    rich_lines = [
        f"# {year} 顶会与 VPWEM 相关论文：图文导读版",
        "",
        f"更新时间：{dt.date.today().isoformat()}",
        "",
        f"共 {len(records)} 篇。图片与摘要素材来自 Connected Researchers；没有站内素材的论文仅保留文字和原文链接。",
        "",
        "说明：本版用于快速阅读与筛选，方法和结果数字在正式引用前仍需回到官方论文核验。",
        "",
    ]
    for venue in VENUE_ORDER:
        selected = [item for item in records if item["venue"] == venue]
        rich_lines.extend([f"## {venue} {year}", "", f"本轮收录 {len(selected)} 篇。", ""])
        if not selected:
            rich_lines.extend(["本轮未发现明确相关论文。", ""])
            continue
        for priority in ("P0", "P1", "P2"):
            group = [item for item in selected if item["priority"] == priority]
            if not group:
                continue
            rich_lines.extend([f"### {priority}", ""])
            for item in group:
                rich_lines.extend([f"#### {item['title']}", ""])
                if item.get("cover_image_path"):
                    rich_lines.extend([f"![{item['title']}]({item['cover_image_path']})", ""])
                links = [f"[导读]({item['guide_url']})"]
                if item["paper_url"]:
                    links.append(f"[原文]({item['paper_url']})")
                if item["github_url"]:
                    links.append(f"[代码]({item['github_url']})")
                rich_lines.append(f"- 导读：{item['guide_text']}")
                if item["method_points"]:
                    rich_lines.append(f"- 方法：{'；'.join(item['method_points'])}")
                if item["result_points"]:
                    rich_lines.append(f"- 结果：{'；'.join(item['result_points'])}")
                rich_lines.extend(
                    [
                        f"- 对 VPWEM：{item['vpwem_relation']}",
                        f"- 链接：{' · '.join(links)}",
                        "",
                    ]
                )
    (output_dir / f"vpwem_{year}_rich_feishu.md").write_text("\n".join(rich_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("literature/vpwem/vpwem_2026_index/vpwem_2026_candidates.json"),
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    source = json.loads(args.index.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(enrich_one, source.get("papers", [])))
    covers_dir = args.index.parent / "assets" / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda record: download_cover(record, covers_dir), records))
    write_outputs(args.index.parent, source, records)
    errors = sum(item["detail_status"] != "ok" for item in records)
    chinese = sum(item["summary_source"] == "中文导读" for item in records)
    covers = sum(item.get("cover_status") == "ok" for item in records)
    print(
        f"Enriched {len(records)} papers: {chinese} Chinese summaries, "
        f"{covers} covers, {errors} detail errors"
    )


if __name__ == "__main__":
    main()
