#!/usr/bin/env python3
"""Download the ICLR 2026 Long Context & Memory collection.

The paper list is obtained from Connected Researchers, while every PDF is
downloaded from the paper's original OpenReview entry. Existing valid PDFs are
kept, so the command is safe to rerun after an interrupted download.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


LIST_ENDPOINT = "https://connectedresearchers.top/api/be/api/guide/iclr/2026"
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
USER_AGENT = "vpwem-literature-downloader/1.0 (academic literature collection)"

# A few accepted-paper titles differ from their arXiv preprint titles.
ARXIV_ALIASES = {
    "GoaWSQWtOE": "2507.19353",  # Smooth Reading (... Long-Context Tasks)
    "eZ5jtFuk3e": "2509.16278",  # Language Modeling with Learned Meta-Tokens
    "GymjF88oGQ": "2602.12108",  # Stateful Language Models Mastering Their Own Context
}


def request_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json,application/pdf", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_paper_list() -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "category": "LLM & Agents",
            "first": "Long Context & Memory",
            "page": 1,
            "limit": 60,
        }
    )
    payload = json.loads(request_bytes(f"{LIST_ENDPOINT}?{query}").decode("utf-8"))
    papers = payload.get("papers", [])
    if payload.get("total") != 49 or len(papers) != 49:
        raise RuntimeError(
            f"Expected 49 papers, received total={payload.get('total')} items={len(papers)}"
        )
    return payload


def openreview_id(forum_url: str) -> str:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(forum_url).query)
    paper_id = query.get("id", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", paper_id):
        raise ValueError(f"Cannot parse OpenReview id from {forum_url!r}")
    return paper_id


def is_pdf(path: Path) -> bool:
    try:
        return path.stat().st_size > 1024 and path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_title(title: str) -> str:
    title = re.sub(r"\\[A-Za-z]+", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return " ".join(title.split())


def arxiv_entries(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    atom = "{http://www.w3.org/2005/Atom}"
    entries = []
    for entry in root.findall(f"{atom}entry"):
        identifier = (entry.findtext(f"{atom}id") or "").rsplit("/", 1)[-1]
        identifier = re.sub(r"v\d+$", "", identifier)
        title = " ".join((entry.findtext(f"{atom}title") or "").split())
        if identifier and title:
            entries.append(
                {
                    "arxiv_id": identifier,
                    "title": title,
                    "pdf_url": f"https://arxiv.org/pdf/{identifier}",
                    "abstract_url": f"https://arxiv.org/abs/{identifier}",
                }
            )
    return entries


def resolve_arxiv_sources(
    papers: list[dict[str, Any]], cache_path: Path, delay: float
) -> dict[str, dict[str, str]]:
    cache: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    for source_id, arxiv_id in ARXIV_ALIASES.items():
        cache[source_id] = {
            "arxiv_id": arxiv_id,
            "title": "title alias resolved manually",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
            "match_score": "alias",
        }
    unresolved = [paper for paper in papers if openreview_id(paper["forum_url"]) not in cache]
    for offset in range(0, len(unresolved), 5):
        batch = unresolved[offset : offset + 5]
        clauses = [f'ti:"{paper["title"].replace(chr(34), " ")}"' for paper in batch]
        query = urllib.parse.urlencode(
            {"search_query": " OR ".join(clauses), "start": 0, "max_results": 15}
        )
        entries: list[dict[str, str]] = []
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                entries = arxiv_entries(request_bytes(f"{ARXIV_ENDPOINT}?{query}", timeout=90))
                last_error = None
                break
            except (OSError, ET.ParseError, urllib.error.URLError) as error:
                last_error = error
                time.sleep(3 * (attempt + 1))
        if last_error:
            print(f"[arxiv api] batch failed: {last_error}", flush=True)
        for paper in batch:
            source_id = openreview_id(paper["forum_url"])
            wanted = normalize_title(paper["title"])
            best: dict[str, str] | None = None
            best_score = 0.0
            for entry in entries:
                score = difflib.SequenceMatcher(None, wanted, normalize_title(entry["title"])).ratio()
                if score > best_score:
                    best, best_score = entry, score
            if best is not None and best_score >= 0.86:
                cache[source_id] = {**best, "match_score": f"{best_score:.4f}"}
                print(f"[arxiv    ] {source_id} -> {best['arxiv_id']}", flush=True)
            else:
                cache[source_id] = {}
                print(f"[no arxiv ] {source_id}  {paper['title']}", flush=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if offset + 5 < len(unresolved):
            time.sleep(max(0.0, delay))
    return cache


def download_one(
    paper: dict[str, Any],
    pdf_dir: Path,
    retries: int,
    arxiv_source: dict[str, str] | None,
) -> dict[str, Any]:
    source_id = openreview_id(paper["forum_url"])
    openreview_pdf = f"https://openreview.net/pdf?id={urllib.parse.quote(source_id)}"
    urls = []
    if arxiv_source and arxiv_source.get("pdf_url"):
        urls.append(("arxiv", arxiv_source["pdf_url"]))
    urls.append(("openreview", openreview_pdf))
    destination = pdf_dir / f"{source_id}.pdf"
    if is_pdf(destination):
        source_name, pdf_url = urls[0]
        return {
            "status": "existing",
            "path": destination,
            "pdf_url": pdf_url,
            "download_source": source_name,
        }

    last_error: Exception | None = None
    last_url = openreview_pdf
    for source_name, pdf_url in urls:
        last_url = pdf_url
        source_retries = retries if source_name == "arxiv" else 1
        for attempt in range(1, source_retries + 1):
            temporary: Path | None = None
            try:
                data = request_bytes(pdf_url, timeout=180)
                if len(data) <= 1024 or not data.startswith(b"%PDF-"):
                    raise RuntimeError(f"Response is not a PDF ({len(data)} bytes)")
                fd, temporary_name = tempfile.mkstemp(prefix=f".{source_id}.", dir=pdf_dir)
                temporary = Path(temporary_name)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                temporary.replace(destination)
                return {
                    "status": "downloaded",
                    "path": destination,
                    "pdf_url": pdf_url,
                    "download_source": source_name,
                }
            except (
                OSError,
                RuntimeError,
                http.client.HTTPException,
                urllib.error.URLError,
            ) as error:
                last_error = error
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                if attempt < source_retries:
                    time.sleep(min(2 ** attempt, 12))
    return {
        "status": "failed",
        "path": destination,
        "pdf_url": last_url,
        "download_source": "none",
        "error": str(last_error),
    }


def extract_text(pdf_path: Path, text_dir: Path) -> str | None:
    if shutil.which("pdftotext") is None:
        return "pdftotext is not installed"
    text_path = text_dir / f"{pdf_path.stem}.txt"
    if text_path.exists() and text_path.stat().st_size > 1000:
        return None
    completed = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(text_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stderr.strip() if completed.returncode else None


def write_manifests(output_dir: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    manifest = {
        "collection": "ICLR 2026 / LLM & Agents / Long Context & Memory",
        "list_source": LIST_ENDPOINT,
        "paper_count": len(rows),
        "papers": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "openreview_id",
        "title",
        "subtopic",
        "forum_url",
        "pdf_url",
        "local_pdf",
        "local_text",
        "sha256",
        "status",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")).replace("\t", " ") for column in columns))
    (output_dir / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "source_response.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("literature/vpwem/iclr2026_long_context_memory"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--arxiv-delay", type=float, default=3.1)
    parser.add_argument("--extract-text", action="store_true")
    args = parser.parse_args()

    output_dir = args.output.resolve()
    pdf_dir = output_dir / "pdfs"
    text_dir = output_dir / "text"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    payload = fetch_paper_list()
    papers = payload["papers"]
    arxiv_sources = resolve_arxiv_sources(
        papers, output_dir / "arxiv_resolution.json", args.arxiv_delay
    )
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_one,
                paper,
                pdf_dir,
                args.retries,
                arxiv_sources.get(openreview_id(paper["forum_url"])),
            ): paper
            for paper in papers
        }
        for future in concurrent.futures.as_completed(futures):
            paper = futures[future]
            source_id = openreview_id(paper["forum_url"])
            try:
                result = future.result()
            except Exception as error:
                source_id = openreview_id(paper["forum_url"])
                result = {
                    "status": "failed",
                    "path": pdf_dir / f"{source_id}.pdf",
                    "pdf_url": "",
                    "download_source": "none",
                    "error": str(error),
                }
            results[source_id] = result
            print(f"[{result['status']:10}] {source_id}  {paper['title']}", flush=True)

    rows: list[dict[str, Any]] = []
    for paper in papers:
        source_id = openreview_id(paper["forum_url"])
        result = results[source_id]
        pdf_path: Path = result["path"]
        text_path = text_dir / f"{source_id}.txt"
        text_error = None
        if args.extract_text and result["status"] != "failed":
            text_error = extract_text(pdf_path, text_dir)
        tags = paper.get("category_tags") or []
        rows.append(
            {
                "openreview_id": source_id,
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "authors": [author.get("name") for author in paper.get("authors", [])],
                "subtopic": tags[-1] if tags else "",
                "category_tags": tags,
                "forum_url": paper.get("forum_url"),
                "pdf_url": result["pdf_url"],
                "download_source": result["download_source"],
                "arxiv": arxiv_sources.get(source_id) or None,
                "local_pdf": str(pdf_path.relative_to(output_dir)),
                "local_text": str(text_path.relative_to(output_dir)) if text_path.exists() else "",
                "sha256": sha256(pdf_path) if is_pdf(pdf_path) else "",
                "status": result["status"],
                "error": result.get("error") or text_error,
                "abstract": paper.get("abstract_raw"),
                "abstract_zh": paper.get("abstract_zh"),
                "guide_analysis": paper.get("guide_analysis"),
                "citation_count": paper.get("s2_citation"),
                "github_url": paper.get("github_url"),
            }
        )

    write_manifests(output_dir, payload, rows)
    failed = [row for row in rows if row["status"] == "failed"]
    print(f"\nCollection: {output_dir}")
    print(f"Valid PDFs: {len(rows) - len(failed)}/{len(rows)}")
    if args.extract_text:
        print(f"Extracted text: {sum(bool(row['local_text']) for row in rows)}/{len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
