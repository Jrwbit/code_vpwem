"""Read WeChat/Alipay exports without optional spreadsheet dependencies.

Amounts are positive integer cents. ``direction`` describes the export's own
classification; refunds, closed trades and internal transfers must be interpreted
by the reconciliation layer, with ``raw`` retaining the source evidence.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from io import StringIO
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile


_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_DIRECTIONS = {"收入": "in", "支出": "out"}
_DATE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def cents(value: str) -> int:
    """Parse an export amount exactly, rejecting fractional cents."""
    value = str(value).strip().replace(",", "").replace("¥", "").replace("￥", "")
    amount = Decimal(value) * 100
    if amount != amount.to_integral_value():
        raise ValueError(f"Amount has fractional cents: {value!r}")
    return abs(int(amount))


def _xlsx_rows(path: str | Path) -> tuple[list[tuple[int, list[str]]], bool]:
    """Read the first worksheet, including sparse cells and shared strings."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(el.itertext()) for el in root.findall("s:si", _NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        properties = workbook.find("s:workbookPr", _NS)
        date1904 = properties is not None and properties.get("date1904") in {"1", "true"}
        sheet = workbook.find("s:sheets/s:sheet", _NS)
        relation_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(el.get("Target") for el in relations if el.get("Id") == relation_id)
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(archive.read(target))
        rows = []
        for row in root.findall("s:sheetData/s:row", _NS):
            values = []
            for cell in row.findall("s:c", _NS):
                letters = re.match(r"[A-Z]+", cell.get("r", "A1")).group()
                index = 0
                for letter in letters:
                    index = index * 26 + ord(letter) - ord("A") + 1
                while len(values) < index:
                    values.append("")
                value_el = cell.find("s:v", _NS)
                value = value_el.text or "" if value_el is not None else ""
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    inline = cell.find("s:is", _NS)
                    value = "".join(inline.itertext()) if inline is not None else ""
                values[index - 1] = value.strip()
            rows.append((int(row.get("r")), values))
        return rows, date1904


def _metadata(source: str, rows: list[tuple[int, list[str]]], header_index: int,
              records: list[dict], encoding: str | None = None) -> dict:
    preamble = "\n".join(" ".join(row) for _, row in rows[:header_index])
    declared = re.search(r"共\s*(\d+)\s*笔记录", preamble)
    totals = {}
    for label, count, amount in re.findall(
        r"(收入|支出|中性交易|不计收支)[：:]\s*(\d+)笔\s*([\d,.]+)元", preamble
    ):
        key = _DIRECTIONS.get(label, "neutral")
        totals[key] = {"count": int(count), "amount": cents(amount)}
    period_line = next((line for line in preamble.splitlines() if "起始时间" in line), "")
    period = _DATE.findall(period_line)
    observed_totals = {
        direction: {"count": sum(record["direction"] == direction for record in records),
                    "amount": sum(record["amount"] for record in records if record["direction"] == direction)}
        for direction in ("in", "out", "neutral")
    }
    mismatches = []
    if declared and int(declared.group(1)) != len(records):
        mismatches.append("declared_count")
    for direction, total in totals.items():
        if total != observed_totals[direction]:
            mismatches.append(f"header_totals.{direction}")
    return {
        "source": source,
        "count": len(records),
        "declared_count": int(declared.group(1)) if declared else None,
        "header_row": rows[header_index][0],
        "header": rows[header_index][1],
        "header_totals": totals,
        "observed_totals": observed_totals,
        "totals_match": not mismatches,
        "mismatches": mismatches,
        "period": {"start": period[0], "end": period[1]} if len(period) >= 2 else None,
        "observed_period": {"start": min(record["time"] for record in records),
                            "end": max(record["time"] for record in records)} if records else None,
        "encoding": encoding,
    }


def _record(raw: dict, row: int, source: str, timestamp: str) -> dict:
    wechat = source == "wechat"
    transaction_id = raw.get("交易单号" if wechat else "交易订单号", "")
    return {
        "id": f"{source}:{row}:{transaction_id}",
        "source": source,
        "source_label": "微信支付" if wechat else "支付宝",
        "row": row,
        "time": timestamp,
        "direction": _DIRECTIONS.get(raw.get("收/支"), "neutral"),
        "amount": cents(raw["金额(元)" if wechat else "金额"]),
        "counterparty": raw.get("交易对方", ""),
        "description": raw.get("商品" if wechat else "商品说明", ""),
        "transaction_type": raw.get("交易类型" if wechat else "交易分类", ""),
        "payment_method": raw.get("支付方式" if wechat else "收/付款方式", ""),
        "status": raw.get("当前状态" if wechat else "交易状态", ""),
        "transaction_id": transaction_id,
        "merchant_id": raw.get("商户单号" if wechat else "商家订单号", ""),
        "raw": raw,
    }


def find_refund_links(records: list[dict]) -> list[dict]:
    """Link Alipay refunds by the exported original-order identifier.

    Both ``*`` and ``_`` introduce suffixes; installment payment records can
    themselves have suffixes. A non-unique order key remains unlinked. WeChat
    export refunds have independent IDs and empty merchant IDs, so this helper
    deliberately makes no amount/time guesses for them.
    """
    originals = defaultdict(list)
    refunds = []
    for record in records:
        if record["source"] != "alipay":
            continue
        if record["status"] == "退款成功":
            refunds.append(record)
        elif record["transaction_id"]:
            originals[re.split(r"[_*]", record["transaction_id"])[0]].append(record)
    links = []
    for refund in refunds:
        key = re.split(r"[_*]", refund["transaction_id"])[0]
        candidates = originals.get(key, [])
        if len(candidates) != 1:
            continue
        original = candidates[0]
        links.append({
            "refund_id": refund["id"],
            "original_id": original["id"],
            "refund_row": refund["row"],
            "original_row": original["row"],
            "amount": refund["amount"],
            "order_key": key,
            "method": "alipay_original_order_id",
        })
    return links


def _reconcile_alipay_metadata(records: list[dict], metadata: dict) -> dict:
    links = find_refund_links(records)
    records_by_id = {record["id"]: record for record in records}
    adjusted = {direction: dict(total) for direction, total in metadata["observed_totals"].items()}
    for record in records:
        if record["direction"] == "in" and record["status"] == "交易关闭":
            adjusted["in"]["amount"] -= record["amount"]
    for link in links:
        original = records_by_id[link["original_id"]]
        adjusted[original["direction"]]["amount"] -= link["amount"]
    adjusted_mismatches = [f"header_totals.{direction}" for direction, total
                           in metadata["header_totals"].items() if total != adjusted[direction]]
    if metadata["declared_count"] is not None and metadata["declared_count"] != len(records):
        adjusted_mismatches.append("declared_count")
    metadata.update({
        "raw_totals_match": metadata["totals_match"],
        "raw_mismatches": list(metadata["mismatches"]),
        "reconciled_totals": adjusted,
        "reconciled_totals_match": not adjusted_mismatches,
        "linked_refund_count": len(links),
        "unmatched_refund_count": sum(record["status"] == "退款成功" for record in records) - len(links),
        "reconciliation_explanation": "收入扣除交易关闭订单；支出及不计收支按原订单扣除关联退款。笔数沿用明细原始笔数。",
    })
    if not adjusted_mismatches:
        metadata["totals_match"] = True
        metadata["mismatches"] = []
        metadata["header_basis"] = "net_after_linked_refunds_and_closed_income"
    elif metadata["raw_totals_match"]:
        metadata["header_basis"] = "raw_totals"
    else:
        metadata["header_basis"] = "unresolved"
    return metadata


def _header_index(rows: list[tuple[int, list[str]]]) -> int:
    for index, (_, row) in enumerate(rows):
        if row and row[0] == "交易时间" and "收/支" in row:
            return index
    raise ValueError("Could not find transaction header: 交易时间 / 收/支")


def parse_wechat(path: str | Path) -> tuple[list[dict], dict]:
    rows, date1904 = _xlsx_rows(path)
    header_index = _header_index(rows)
    header = rows[header_index][1]
    records = []
    for row_number, values in rows[header_index + 1:]:
        if not values or not any(values):
            continue
        raw = {key: values[index] if index < len(values) else ""
               for index, key in enumerate(header) if key}
        timestamp = raw["交易时间"]
        if not _DATE.fullmatch(timestamp):
            seconds = int((Decimal(timestamp) * 86400).to_integral_value(rounding=ROUND_HALF_UP))
            epoch = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
            timestamp = (epoch + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        records.append(_record(raw, row_number, "wechat", timestamp))
    return records, _metadata("wechat", rows, header_index, records)


def parse_alipay(path: str | Path) -> tuple[list[dict], dict]:
    data = Path(path).read_bytes()
    try:
        content, encoding = data.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        content, encoding = data.decode("gb18030"), "gb18030"
    reader = csv.reader(StringIO(content, newline=""))
    rows = []
    for values in reader:
        rows.append((reader.line_num, [value.strip() for value in values]))
    header_index = _header_index(rows)
    header = rows[header_index][1]
    records = []
    for row_number, values in rows[header_index + 1:]:
        if not values or not any(values):
            continue
        raw = {key: values[index] if index < len(values) else ""
               for index, key in enumerate(header) if key}
        timestamp = raw["交易时间"]
        datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        records.append(_record(raw, row_number, "alipay", timestamp))
    metadata = _metadata("alipay", rows, header_index, records, encoding)
    return records, _reconcile_alipay_metadata(records, metadata)
