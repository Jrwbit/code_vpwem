#!/usr/bin/env python3
"""Build a coarse yearly VPWEM-related paper index from Connected Researchers."""

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


API_ROOT = "https://connectedresearchers.top/api/be/api/guide"
GUIDE_ROOT = "https://connectedresearchers.top/guide"
VENUES = ("iclr", "icml", "cvpr", "aaai", "wacv", "aamas", "acl", "eacl", "wsdm")
VENUE_NAMES = {
    "iclr": "ICLR",
    "icml": "ICML",
    "cvpr": "CVPR",
    "aaai": "AAAI",
    "wacv": "WACV",
    "aamas": "AAMAS",
    "acl": "ACL",
    "eacl": "EACL",
    "wsdm": "WSDM",
}

ROBOT_TERMS = (
    "robot", "robotic", "manipulation", "visuomotor", "vision-language-action",
    "vision language action", "vla", "imitation learning", "embodied agent",
    "embodied ai", "motor policy", "control policy", "dexterous",
)
ROBOT_STRONG_TITLE_TERMS = (
    "robot", "robotic", "visuomotor", "vision-language-action",
    "vision language action", "vla", "embodied", "dexterous",
)
MEMORY_TERMS = (
    "memory", "memorization", "remember", "history", "historical", "recurrent",
    "non-markov", "non markov", "partially observable", "partial observability",
    "pomdp", "long-term", "long term", "long-horizon", "long horizon",
    "temporal context", "temporal-aware", "temporal aware", "context compression",
    "streaming", "task progress", "action history", "past observation",
)
POLICY_TERMS = (
    "diffusion policy", "action policy", "action generation", "action chunk",
    "latent action", "flow policy", "policy learning", "behavior cloning",
    "behaviour cloning", "trajectory", "world model", "reasoning policy",
)
TRANSFER_TERMS = (
    "episodic memory", "working memory", "memory compression", "token compression",
    "long context", "long-context", "infinite context", "video memory",
    "selective memory", "memory retrieval", "memory augmented", "memory-augmented",
)
TEMPORAL_TITLE_TERMS = (
    "memory", "memorization", "history", "historical", "recurrent", "remember",
    "long-term", "long term", "long-horizon", "long horizon", "non-markov",
    "non markov", "partial observability", "partially observable", "pomdp",
    "temporal context", "temporal-aware", "temporal aware", "task progress",
    "progress-aware", "progress aware", "streaming", "past observation",
)
DIRECT_MEMORY_TITLE_TERMS = (
    "memory", "memorization", "history", "historical", "recurrent", "remember",
    "non-markov", "non markov", "partial observability", "partially observable", "pomdp",
)
ROBOT_METHOD_TITLE_TERMS = (
    "diffusion policy", "action chunk", "latent action", "world model",
    "trajectory", "visuomotor policy", "policy learning", "flow policy",
)
TRANSFER_TITLE_TERMS = (
    "episodic memory", "working memory", "memory compression", "context compression",
    "token compression", "selective memory",
    "video memory", "streaming video", "memory experts", "memory-augmented diffusion",
)
GENERIC_LANGUAGE_TERMS = (
    "language model", "llm", "rag", "retrieval-augmented generation", "kv cache",
    "kvcache", "text-to-sql", "coding agent", "web agent",
)
EXCLUDE_TERMS = (
    "memory footprint", "gpu memory", "out-of-memory", "memory-efficient training",
    "memory efficient training", "memory safety", "cache replacement",
)
EXCLUDE_TITLES = (
    "streaming drag-oriented interactive video manipulation",
    "from passive perception to active memory: a weakly supervised image manipulation",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_any(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def request_json(url: str, retries: int = 5) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 VPWEM-index/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_page(venue: str, year: int, page: int, limit: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"page": page, "limit": limit})
    payload = request_json(f"{API_ROOT}/{venue}/{year}?{query}")
    return payload.get("papers", [])


def fetch_venue(venue: str, year: int, workers: int, limit: int) -> list[dict[str, Any]]:
    first = request_json(f"{API_ROOT}/{venue}/{year}?page=1&limit={limit}")
    total = int(first.get("total", 0))
    pages = (total + limit - 1) // limit
    papers = list(first.get("papers", []))
    if pages > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(fetch_page, venue, year, page, limit)
                for page in range(2, pages + 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                papers.extend(future.result())
    by_id = {paper["paper_id"]: paper for paper in papers if paper.get("paper_id")}
    if len(by_id) != total:
        raise RuntimeError(f"{venue}: expected {total} papers, received {len(by_id)} unique records")
    return list(by_id.values())


def score_paper(paper: dict[str, Any]) -> dict[str, Any] | None:
    title = paper.get("title", "")
    abstract = paper.get("abstract_raw", "") or paper.get("abstract_zh", "")
    guide = paper.get("guide_analysis") or {}
    guide_text = " ".join(
        str(value)
        for key, value in guide.items()
        if key in {"problem_en", "one_liner_en", "method_en"}
    )
    title_text = normalize(title)
    if any(value in title_text for value in EXCLUDE_TITLES):
        return None
    text = normalize(f"{title} {abstract} {guide_text}")
    robot = contains_any(text, ROBOT_TERMS)
    memory = contains_any(text, MEMORY_TERMS)
    policy = contains_any(text, POLICY_TERMS)
    transfer = contains_any(text, TRANSFER_TERMS)
    excluded = contains_any(text, EXCLUDE_TERMS)
    title_robot = contains_any(title_text, ROBOT_TERMS)
    title_strong_robot = contains_any(title_text, ROBOT_STRONG_TITLE_TERMS)
    title_temporal = contains_any(title_text, TEMPORAL_TITLE_TERMS)
    title_direct_memory = contains_any(title_text, DIRECT_MEMORY_TITLE_TERMS)
    title_policy = contains_any(title_text, ROBOT_METHOD_TITLE_TERMS)
    title_transfer = contains_any(title_text, TRANSFER_TITLE_TERMS)
    generic_language = contains_any(title_text, GENERIC_LANGUAGE_TERMS)

    # The index intentionally favors title-level evidence. Abstract-only matches
    # created hundreds of generic long-task and LLM-memory false positives.
    robot_manipulation = "manipulation" in title_text and bool(robot)
    if (title_strong_robot or robot_manipulation) and title_direct_memory:
        level = "P0"
        relation = "机器人策略中的记忆/历史"
        score = 86 + min(9, len(title_direct_memory) * 3)
    elif (title_strong_robot or robot_manipulation) and (title_temporal or title_policy):
        level = "P1"
        relation = "长时程机器人策略/动作建模"
        score = 72 + min(10, (len(title_temporal) + len(title_policy)) * 2)
    elif title_transfer and not excluded and not generic_language:
        level = "P2"
        relation = "可迁移的记忆压缩方法"
        score = 60 + min(10, len(title_transfer) * 2)
    elif (
        any(term in title_text for term in ("long-context", "long context"))
        and any(term in title_text for term in ("visual", "video", "multimodal", "memory", "compression"))
        and not generic_language
    ):
        level = "P2"
        relation = "可迁移的视觉记忆方法"
        score = 58 + min(10, len(title_transfer) * 2)
    else:
        return None

    paper_id = paper.get("paper_id", "")
    forum_url = paper.get("forum_url") or ""
    return {
        "venue": VENUE_NAMES.get(paper.get("venue_code", ""), paper.get("venue_code", "").upper()),
        "paper_id": paper_id,
        "title": title,
        "priority": level,
        "relation": relation,
        "score": score,
        "guide_url": f"{GUIDE_ROOT}/paper/{urllib.parse.quote(paper_id, safe='')}",
        "paper_url": forum_url,
        "github_url": paper.get("github_url") or "",
        "matched_terms": sorted(set(robot + memory + policy + transfer)),
    }


def write_outputs(
    output_dir: Path,
    year: int,
    records: list[dict[str, Any]],
    scanned: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda item: (VENUES.index(item["venue"].lower()), -item["score"], item["title"]))
    payload = {
        "year": year,
        "source": GUIDE_ROOT,
        "scanned": scanned,
        "candidate_count": len(records),
        "papers": records,
    }
    (output_dir / f"vpwem_{year}_candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = ("venue", "priority", "title", "relation", "guide_url", "paper_url", "github_url")
    with (output_dir / f"vpwem_{year}_papers.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        f"# {year} 顶会中与 VPWEM 相关的论文索引",
        "",
        f"更新时间：{dt.date.today().isoformat()}",
        "",
        "用途：先建立可阅读链接库；本版只做粗筛，不替代后续逐篇全文核验。",
        "",
        f"扫描范围：{len(scanned)} 个会议，共 {sum(scanned.values()):,} 篇；保留 {len(records)} 篇候选。",
        "",
        "优先级：P0 为直接相关，P1 为重要邻近工作，P2 为可迁移方法。",
        "",
    ]
    for venue in VENUES:
        name = VENUE_NAMES[venue]
        venue_records = [item for item in records if item["venue"] == name]
        lines.extend(
            [
                f"## {name} {year}",
                "",
                f"候选 {len(venue_records)} 篇；会议入口：{GUIDE_ROOT}/{venue}/{year}",
                "",
            ]
        )
        if not venue_records:
            lines.extend(["本轮粗筛未发现明确相关论文。", ""])
            continue
        for index, item in enumerate(venue_records, 1):
            links = [f"[导读]({item['guide_url']})"]
            if item["paper_url"]:
                links.append(f"[原文]({item['paper_url']})")
            if item["github_url"]:
                links.append(f"[代码]({item['github_url']})")
            lines.extend(
                [
                    f"### {index}. {item['title']}",
                    "",
                    f"- 优先级：{item['priority']}",
                    f"- 关系：{item['relation']}",
                    f"- 链接：{' · '.join(links)}",
                    "",
                ]
            )
    (output_dir / f"vpwem_{year}_feishu.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=60)
    parser.add_argument(
        "--refine-existing",
        type=Path,
        help="Re-score an existing candidate JSON without downloading the conference indexes again.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path(f"literature/vpwem/vpwem_{args.year}_index")

    if args.refine_existing:
        payload = json.loads(args.refine_existing.read_text(encoding="utf-8"))
        records = []
        for item in payload.get("papers", []):
            source = {
                "venue_code": item["venue"].lower(),
                "paper_id": item["paper_id"],
                "title": item["title"],
                "abstract_raw": " ".join(item.get("matched_terms", [])),
                "forum_url": item.get("paper_url", ""),
                "github_url": item.get("github_url", ""),
            }
            record = score_paper(source)
            if record:
                records.append(record)
        year = int(payload.get("year", args.year))
        write_outputs(output_dir, year, records, payload.get("scanned", {}))
        print(f"Refined {len(payload.get('papers', []))} candidates to {len(records)}")
        return

    scanned: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for venue in VENUES:
        papers = fetch_venue(venue, args.year, args.workers, args.page_size)
        scanned[VENUE_NAMES[venue]] = len(papers)
        for paper in papers:
            record = score_paper(paper)
            if record:
                records.append(record)
        print(f"{VENUE_NAMES[venue]}: scanned {len(papers)}, candidates {sum(r['venue'] == VENUE_NAMES[venue] for r in records)}")
    write_outputs(output_dir, args.year, records, scanned)


if __name__ == "__main__":
    main()
