"""Parse bank PDF exports into auditable, integer-cent transaction records.

Requires PyMuPDF for table geometry and watermark filtering. Input PDFs and
extracted text remain outside this repository; this module never writes them.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple


class BankStatementError(ValueError):
    """A statement could not be read or reconciled without guessing."""


class _CellBox(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float


def money_cents(value: str) -> int:
    """Convert a printed currency value exactly; reject fractional cents."""
    clean = value.strip().replace(",", "").replace("，", "")
    clean = clean.replace("￥", "").replace("¥", "").replace("−", "-")
    if clean.startswith("(") and clean.endswith(")"):
        clean = "-" + clean[1:-1]
    try:
        scaled = Decimal(clean) * 100
    except InvalidOperation as exc:
        raise BankStatementError(f"Invalid amount: {value!r}") from exc
    if not scaled.is_finite() or scaled != scaled.to_integral_value():
        raise BankStatementError(f"Amount must have whole cents: {value!r}")
    return int(scaled)


def extract_pdf_pages(path: str | Path, password: str | None = None) -> list[str]:
    """Read all PDF pages in layout order, preserving page boundaries."""
    command = ["pdftotext", "-layout", "-enc", "UTF-8"]
    if password is not None:
        command += ["-upw", password]
    command += [str(Path(path)), "-"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        # Do not construct CalledProcessError: it embeds password-bearing argv.
        if "password" in error.lower():
            raise BankStatementError("PDF password is missing or incorrect")
        raise BankStatementError(f"PDF text extraction failed: {error}")
    text = result.stdout.decode("utf-8")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages or not any(page.strip() for page in pages):
        raise BankStatementError("PDF contains no extractable text")
    return pages


def _audit(records: list[dict[str, Any]], page_count: int) -> dict[str, Any]:
    page_totals = [
        {"page": page, "rows": 0, "income_cents": 0, "expense_cents": 0}
        for page in range(1, page_count + 1)
    ]
    by_currency = {}
    for record in records:
        page = page_totals[record["raw"]["page"] - 1]
        page["rows"] += 1
        currency = record.get("currency", "CNY")
        total = by_currency.setdefault(currency, {"rows": 0, "income_cents": 0, "expense_cents": 0})
        total["rows"] += 1
        field = {"in": "income_cents", "out": "expense_cents"}.get(record["direction"])
        if field:
            total[field] += record["amount"]
            if currency == "CNY":
                page[field] += record["amount"]
    return {
        "row_count": len(records),
        "page_count": page_count,
        "page_totals": page_totals,
        "income_cents": sum(p["income_cents"] for p in page_totals),
        "expense_cents": sum(p["expense_cents"] for p in page_totals),
        "currency": "CNY",
        "totals_by_currency": by_currency,
        "first_transaction": min((r["time"] for r in records), default=None),
        "last_transaction": max((r["time"] for r in records), default=None),
    }


def _horizontal_lines(page: Any) -> list[dict[str, Any]]:
    """Discard diagonal watermark text before assigning text to table cells."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if abs(line["dir"][1]) > 0.01:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append({"bbox": line["bbox"], "text": text})
    return lines


def _cell_text(lines: list[dict[str, Any]], box: Any) -> str:
    selected = []
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        if box.x0 <= (x0 + x1) / 2 < box.x1 and box.y0 <= (y0 + y1) / 2 < box.y1:
            selected.append(line)
    return "".join(line["text"] for line in sorted(selected, key=lambda line: (line["bbox"][1], line["bbox"][0]))).strip()


def _horizontal_characters(page: Any) -> list[dict[str, Any]]:
    # BOC emits some adjacent cells as one PDF text line; character bounds keep
    # the remark and counterparty columns separate without splitting on spaces.
    return [
        {"bbox": character["bbox"], "text": character["c"]}
        for block in page.get_text("rawdict")["blocks"]
        for line in block.get("lines", []) if abs(line["dir"][1]) < 0.01
        for span in line["spans"]
        for character in span["chars"]
    ]


def _validate_balances(records: list[dict[str, Any]], descending: bool = False) -> int:
    ordered = list(reversed(records)) if descending else records
    previous_by_currency = {}
    checks = 0
    for current in ordered:
        currency = current.get("currency", "CNY")
        previous = previous_by_currency.get(currency)
        if previous:
            delta = current["amount"] * (1 if current["direction"] == "in" else -1)
            if previous["raw"]["balance_cents"] + delta != current["raw"]["balance_cents"]:
                raise BankStatementError(f"Balance discontinuity ({currency}): {previous['id']} -> {current['id']}")
            checks += 1
        previous_by_currency[currency] = current
    return checks


def _parse_cmb(document: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse CMB's six-column date-only statement with variable-height rows."""
    records = []
    first_text = "\n".join(line["text"] for line in _horizontal_lines(document[0]))
    account = re.search(r"账号：(\d+)", first_text).group(1)
    period = re.search(r"(\d{4}-\d{2}-\d{2}) -- (\d{4}-\d{2}-\d{2})", first_text)
    for page_number, page in enumerate(document, 1):
        lines = _horizontal_lines(page)
        dates = sorted((line for line in lines if line["bbox"][0] < 80 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", line["text"])), key=lambda line: line["bbox"][1])
        centers = [(line["bbox"][1] + line["bbox"][3]) / 2 for line in dates]
        if not centers:
            raise BankStatementError(f"CMB page {page_number} contains no date rows")
        boundaries = [centers[0] - 13.5] + [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [centers[-1] + 20]
        xs = [36, 98.73, 156.26, 234.71, 307.87, 417.63, 560]
        for page_row, (top, bottom) in enumerate(zip(boundaries, boundaries[1:]), 1):
            cells = [_cell_text(lines, _CellBox(left, top, right, bottom)) for left, right in zip(xs, xs[1:])]
            if cells[1] != "CNY":
                raise BankStatementError(f"CMB page {page_number}: unsupported currency {cells[1]!r}")
            signed = money_cents(cells[2])
            counterparty = re.sub(r"\s*\d{5,}$", "", cells[5]).strip()
            records.append({
                "id": f"bank_cmb:{len(records) + 1}", "source": "bank_cmb", "source_label": "招商银行",
                "bank_name": "招商银行", "account_tail": account[-4:], "row": len(records) + 1,
                "currency": "CNY", "currency_label": "人民币",
                "time": datetime.strptime(cells[0], "%Y-%m-%d").strftime("%Y-%m-%d 00:00:00"),
                "time_precision": "day", "direction": "in" if signed > 0 else "out" if signed < 0 else "neutral",
                "amount": abs(signed), "counterparty": counterparty, "description": cells[4],
                "payment_method": f"招商银行储蓄卡({account[-4:]})", "status": "已入账",
                "transaction_id": "", "merchant_id": "",
                "raw": {"page": page_number, "page_row": page_row, "account": account, "summary": cells[4],
                        "balance_cents": money_cents(cells[3]), "counterparty_info": cells[5], "cells": cells},
            })
    checks = _validate_balances(records)
    return records, {
        "source": "bank_cmb", "source_label": "招商银行", "account_last4": account[-4:],
        "period_start": period.group(1) if period else None, "period_end": period.group(2) if period else None,
        "printed_totals": None, "printed_totals_match": None, "balance_checks": checks, "balance_discontinuities": [],
        "notes": ["账单未印收支汇总；逐页日期行全量提取，逐笔余额连续校验通过。", "仅提供记账日期，无时分秒；同日匹配需结合金额和对手。", "此来源覆盖至10月20日，晚于该日的空白不代表无交易。"],
        **_audit(records, len(document)),
    }


def _parse_boc(document: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse BOC's twelve-column ruled statement and validate every page."""
    records = []
    printed_pages = []
    first_text = "\n".join(line["text"] for line in _horizontal_lines(document[0]))
    account = re.search(r"借记卡号：(\d+)", first_text).group(1)
    period = re.search(r"交易区间：(\d{4}-\d{2}-\d{2}) 至(\d{4}-\d{2}-\d{2})", first_text)
    for page_number, page in enumerate(document, 1):
        lines = _horizontal_lines(page)
        text = "\n".join(line["text"] for line in lines)
        debit = re.search(r"借方发生数：\s*([\d,.]+)", text)
        credit = re.search(r"贷方发生数：\s*([\d,.]+)", text)
        row_count = re.search(r"行数:\s*(\d+)", text)
        if not all((debit, credit, row_count)):
            raise BankStatementError(f"BOC page {page_number}: printed totals missing")
        printed = {"page": page_number, "rows": int(row_count.group(1)), "expense_cents": money_cents(debit.group(1)), "income_cents": money_cents(credit.group(1))}
        xs, ys = set(), set()
        for drawing in page.get_drawings():
            for item in drawing["items"]:
                if item[0] != "l":
                    continue
                a, b = item[1:]
                if abs(a.x - b.x) < 0.1 and min(a.y, b.y) < 137 < max(a.y, b.y):
                    xs.add(round(a.x, 2))
                if abs(a.y - b.y) < 0.1 and min(a.x, b.x) < 64 < max(a.x, b.x):
                    ys.add(round(a.y, 2))
        xs, ys = sorted(xs), sorted(ys)
        if len(xs) != 13 or len(ys) != printed["rows"] + 2:
            raise BankStatementError(f"BOC page {page_number}: table grid does not match printed row count")
        chars = _horizontal_characters(page)
        page_records = []
        for page_row, (top, bottom) in enumerate(zip(ys[1:], ys[2:]), 1):
            cells = [_cell_text(chars, _CellBox(left, top, right, bottom)) for left, right in zip(xs, xs[1:])]
            cells = ["" if re.fullmatch(r"-+", cell) else cell for cell in cells]
            currency = {"人民币": "CNY", "美元": "USD", "港币": "HKD", "印度尼西亚卢比": "IDR"}.get(cells[2])
            if not currency:
                raise BankStatementError(f"BOC page {page_number}: unsupported currency {cells[2]!r}")
            signed = money_cents(cells[3])
            record = {
                "id": f"bank_boc:{len(records) + 1}", "source": "bank_boc", "source_label": "中国银行",
                "bank_name": "中国银行", "account_tail": account[-4:], "row": len(records) + 1,
                "currency": currency, "currency_label": cells[2],
                "time": datetime.strptime(cells[0] + " " + cells[1], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S"),
                "time_precision": "second", "direction": "in" if signed > 0 else "out" if signed < 0 else "neutral",
                "amount": abs(signed), "counterparty": cells[9], "description": cells[5] + ("；" + cells[8] if cells[8] else ""),
                "payment_method": f"中国银行储蓄卡({account[-4:]})", "status": "已入账",
                "transaction_id": "", "merchant_id": "",
                "raw": {"page": page_number, "page_row": page_row, "account": account, "summary": cells[5],
                        "balance_cents": money_cents(cells[4]), "channel": cells[6], "branch": cells[7], "remark": cells[8],
                        "counterparty_account": cells[10], "counterparty_bank": cells[11], "cells": cells},
            }
            records.append(record)
            page_records.append(record)
        actual = {"page": page_number, "rows": len(page_records), "expense_cents": sum(r["amount"] for r in page_records if r["direction"] == "out"), "income_cents": sum(r["amount"] for r in page_records if r["direction"] == "in")}
        if printed != actual:
            raise BankStatementError(f"BOC page {page_number}: printed totals mismatch {printed} != {actual}")
        printed_pages.append(printed)
    checks = _validate_balances(records, descending=True)
    return records, {
        "source": "bank_boc", "source_label": "中国银行", "account_last4": account[-4:],
        "period_start": period.group(1) if period else None, "period_end": period.group(2) if period else None,
        "printed_page_totals": printed_pages, "printed_totals_match": True, "balance_checks": checks, "balance_discontinuities": [],
        "printed_totals_scope": "各币种票面数字相加，仅作为提取完整性校验，不是人民币收支合计",
        "notes": ["各页行数、借方发生数和贷方发生数均与提取结果一致；按币种逐笔余额连续校验通过。", "包含人民币、美元、港币、印度尼西亚卢比；收入支出主汇总仅含人民币，外币独立保留。", "原文件按日期及币种排列；记账时间可能与实际交易提交时间有差异。"],
        **_audit(records, len(document)),
    }


def _parse_cib(document: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse Industrial Bank's ruled nine-column account statement."""
    records = []
    first_lines = _horizontal_lines(document[0])
    first_text = "\n".join(line["text"] for line in first_lines)
    header = re.search(r"支出\s*[¥￥]([\d,.]+)\((\d+)\)笔\s*收入\s*[¥￥]([\d,.]+)\((\d+)\)笔", first_text)
    if not header:
        raise BankStatementError("Industrial Bank's printed totals are missing")
    period = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})", first_text)
    account_match = re.search(r"账户账号：(\d+)", first_text)
    account = account_match.group(1) if account_match else ""
    printed = {
        "expense_cents": money_cents(header.group(1)),
        "expense_count": int(header.group(2)),
        "income_cents": money_cents(header.group(3)),
        "income_count": int(header.group(4)),
    }
    for page_number, page in enumerate(document, 1):
        lines = _horizontal_lines(page)
        rectangles = [drawing["rect"] for drawing in page.get_drawings()]
        first_cells = sorted(
            (box for box in rectangles if abs(box.x0 - 50) < 1 and abs(box.x1 - 100.86) < 1),
            key=lambda box: box.y0,
        )
        page_row = 0
        for first_cell in first_cells:
            date_text = _cell_text(lines, first_cell)
            if not re.match(r"^\d{4}-\d{2}-\d{2}", date_text):
                continue
            row_edges = sorted(
                (box for box in rectangles if abs(box.y0 - first_cell.y0) < 0.1 and abs(box.y1 - first_cell.y1) < 0.1),
                key=lambda box: box.x1,
            )
            if len(row_edges) != 9:
                raise BankStatementError(f"Industrial Bank page {page_number}: expected 9 cells")
            boundaries = [first_cell.x0] + [edge.x1 for edge in row_edges]
            cell_boxes = [type(first_cell)(left, first_cell.y0, right, first_cell.y1) for left, right in zip(boundaries, boundaries[1:])]
            cells = [_cell_text(lines, box) for box in cell_boxes]
            time = datetime.strptime(cells[0], "%Y-%m-%d%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            signed_amount = money_cents(cells[4])
            direction = {"收": "in", "支": "out"}.get(cells[3])
            if direction is None or (signed_amount < 0) != (direction == "out"):
                raise BankStatementError(f"Industrial Bank page {page_number}: amount/direction mismatch")
            order = re.match(r"订单(\d+)", cells[6])
            page_row += 1
            records.append({
                "id": f"bank_cib:{len(records) + 1}",
                "source": "bank_cib",
                "source_label": "兴业银行",
                "bank_name": "兴业银行",
                "account_tail": account[-4:],
                "currency": "CNY", "currency_label": "人民币",
                "row": len(records) + 1,
                "time": time,
                "time_precision": "second",
                "direction": direction,
                "amount": abs(signed_amount),
                "counterparty": cells[7],
                "description": cells[2] + ("；" + cells[6] if cells[6] else ""),
                "payment_method": f"兴业银行储蓄卡({account[-4:]})",
                "status": "已入账",
                "transaction_id": "",
                "merchant_id": order.group(1) if order else "",
                "raw": {
                    "page": page_number,
                    "page_row": page_row,
                    "account": account,
                    "booking_date": cells[1],
                    "summary": cells[2],
                    "usage": cells[6],
                    "balance_cents": money_cents(cells[5]),
                    "counterparty_account_bank": cells[8],
                    "cells": cells,
                },
            })
        # Independently count date anchors to detect a missing/malformed table row.
        date_count = sum(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", line["text"])) and line["bbox"][0] < 100 for line in lines)
        if page_row != date_count:
            raise BankStatementError(f"Industrial Bank page {page_number}: parsed {page_row}/{date_count} date rows")
    audit = _audit(records, len(document))
    actual = {
        "expense_cents": audit["expense_cents"],
        "income_cents": audit["income_cents"],
        "expense_count": sum(r["direction"] == "out" for r in records),
        "income_count": sum(r["direction"] == "in" for r in records),
    }
    if actual != printed:
        raise BankStatementError(f"Industrial Bank printed totals mismatch: {actual} != {printed}")
    balance_checks = []
    for previous, current in zip(records, records[1:]):
        delta = current["amount"] * (1 if current["direction"] == "in" else -1)
        if previous["raw"]["balance_cents"] + delta != current["raw"]["balance_cents"]:
            balance_checks.append({"previous_id": previous["id"], "current_id": current["id"]})
    if balance_checks:
        raise BankStatementError(f"Industrial Bank balance discontinuities: {balance_checks}")
    return records, {
        "source": "bank_cib", "source_label": "兴业银行", "account_last4": account[-4:],
        "period_start": period.group(1) if period else None,
        "period_end": period.group(2) if period else None,
        "printed_totals": printed, "printed_totals_match": True,
        "balance_checks": len(records) - 1, "balance_discontinuities": [],
        "notes": ["金额与首页收支笔数/总额一致；逐笔余额连续校验通过；已按文字方向过滤斜向水印。"],
        **audit,
    }


def parse_bank_pdf(
    path: str | Path, password: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse a supported PDF; reject unknown formats rather than losing rows."""
    try:
        import pymupdf
    except ImportError as exc:
        raise BankStatementError("Bank PDF parsing requires PyMuPDF (pip install pymupdf)") from exc
    with pymupdf.open(path) as document:
        if document.needs_pass and (password is None or not document.authenticate(password)):
            raise BankStatementError("PDF password is missing or incorrect")
        first_text = "\n".join(line["text"] for line in _horizontal_lines(document[0]))
        if "兴业银行交易明细" in first_text:
            return _parse_cib(document)
        if "招商银行交易流水" in first_text:
            return _parse_cmb(document)
        if "中国银行交易流水明细清单" in first_text:
            return _parse_boc(document)
        raise BankStatementError("This bank PDF layout has not been identified")
