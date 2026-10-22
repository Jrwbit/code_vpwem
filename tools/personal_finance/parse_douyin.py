"""Read Douyin's PDF payment statement using Poppler word coordinates."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET

from .parse_apps import cents


_NS = {"h": "http://www.w3.org/1999/xhtml"}
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_HEADERS = ("交易时间", "收/支/其他", "金额（元）", "收/付款方式", "交易单号", "交易对方", "商家单号")


def parse_douyin_bbox(xml: str | bytes) -> tuple[list[dict], dict]:
    """Parse ``pdftotext -bbox-layout`` output, joining wrapped cell contents."""
    root = ET.fromstring(xml)
    pages = root.findall(".//h:page", _NS)
    records = []
    page_counts = []
    period = None
    for page_number, page in enumerate(pages, 1):
        words = [{"text": word.text or "", "x": (float(word.get("xMin")) + float(word.get("xMax"))) / 2,
                  "y": float(word.get("yMin"))}
                 for word in page.findall(".//h:word", _NS)]
        words.sort(key=lambda word: (word["y"], word["x"]))
        if page_number == 1:
            text = " ".join(word["text"] for word in words)
            found = re.search(r"交易时间段：\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+至\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text)
            if found:
                period = {"start": found.group(1), "end": found.group(2)}
        header = [next((word for word in words if word["text"] == label), None) for label in _HEADERS]
        if not all(header):
            if any(_DATE.fullmatch(word["text"]) for word in words):
                raise ValueError(f"Unrecognized Douyin table header on page {page_number}")
            page_counts.append({"page": page_number, "count": 0})
            continue
        boundaries = [(left["x"] + right["x"]) / 2 for left, right in zip(header, header[1:])]
        anchors = [word for word in words if _DATE.fullmatch(word["text"])
                   and word["x"] < boundaries[0] and word["y"] > header[0]["y"]]
        footer_y = min((word["y"] for word in words if re.fullmatch(r"\d+/|\d+/\d+", word["text"])),
                       default=float(page.get("height")) - 28)
        for index, anchor in enumerate(anchors):
            next_y = anchors[index + 1]["y"] if index + 1 < len(anchors) else footer_y
            cells = [[] for _ in _HEADERS]
            for word in words:
                if not anchor["y"] - 0.1 <= word["y"] < next_y - 0.1:
                    continue
                # A footer has a centered fraction instead of table cell data.
                if re.fullmatch(r"\d+/|\d+/\d+", word["text"]):
                    continue
                cells[bisect_right(boundaries, word["x"])].append(word["text"])
            values = [" ".join(cells[0])] + ["".join(cell) for cell in cells[1:]]
            raw = dict(zip(_HEADERS, values))
            raw.update({"page": page_number, "page_row": index + 1})
            timestamp, direction, amount, method, order, counterparty, merchant = values
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            if direction not in {"收入", "支出", "其他"}:
                raise ValueError(f"Unrecognized Douyin direction on page {page_number}: {direction!r}")
            if not re.fullmatch(r"\d+", order):
                raise ValueError(f"Invalid Douyin order identifier on page {page_number}")
            is_refund = method == "退款"
            records.append({
                "id": f"douyin:{page_number}:{index + 1}:{order}",
                "source": "douyin", "source_label": "抖音支付", "row": len(records) + 1,
                "time": timestamp, "direction": {"收入": "in", "支出": "out", "其他": "neutral"}[direction],
                "amount": cents(amount), "counterparty": counterparty,
                "description": f"{counterparty}{'退款' if is_refund else ''}",
                "transaction_type": "退款" if is_refund else ("其他" if direction == "其他" else "支付"),
                "payment_method": method, "status": "", "transaction_id": order,
                "merchant_id": merchant, "raw": raw,
            })
        page_counts.append({"page": page_number, "count": len(anchors)})
    if not records:
        raise ValueError("No Douyin payment rows found")
    totals = {direction: {"count": sum(record["direction"] == direction for record in records),
                          "amount": sum(record["amount"] for record in records if record["direction"] == direction)}
              for direction in ("in", "out", "neutral")}
    return records, {
        "source": "douyin", "count": len(records), "declared_count": None,
        "page_count": len(pages), "page_counts": page_counts, "header": list(_HEADERS),
        "header_totals": {}, "observed_totals": totals, "totals_match": None,
        "audit_note": "PDF无汇总笔数和总额；已按每页日期锚点核对提取行数，无法核对官方汇总。",
        "period": period,
        "observed_period": {"start": min(record["time"] for record in records), "end": max(record["time"] for record in records)},
    }


def parse_douyin(path: str | Path) -> tuple[list[dict], dict]:
    result = subprocess.run(["pdftotext", "-bbox-layout", "-enc", "UTF-8", str(path), "-"],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Douyin PDF text extraction failed: " + result.stderr.decode("utf-8", errors="replace").strip())
    return parse_douyin_bbox(result.stdout)
