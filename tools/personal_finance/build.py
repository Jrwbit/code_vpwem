"""Build an offline personal-finance dashboard from already decrypted exports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime
from decimal import Decimal
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

from .douyin_audit import annotate_douyin_chains
from .foreign_audit import annotate_foreign_cash
from .entities import annotate_entities
from .parse_apps import parse_alipay, parse_wechat
from .parse_banks import parse_bank_pdf
from .parse_douyin import parse_douyin
from .reconcile import KINDS, fingerprint, monthly_totals, reconcile
from .self_transfer_audit import audit_self_transfers
from .spending_categories import annotate_spending_categories


SOURCE_LABELS = {
    "bank_cib": "兴业银行", "bank_cmb": "招商银行", "bank_boc": "中国银行",
    "wechat": "微信支付", "alipay": "支付宝", "douyin": "抖音支付",
}
_EMBED_MARKER = "/*__FINANCE_DATA__*/ null"


def json_for_script(data: dict) -> str:
    """Keep untrusted merchant text inside the JSON expression in a script."""
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_dashboard(data: dict, template: str | None = None) -> str:
    if template is None:
        template = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
    if template.count(_EMBED_MARKER) != 1:
        raise ValueError("Dashboard template must contain exactly one finance-data marker")
    return template.replace(_EMBED_MARKER, json_for_script(data), 1)


def _input_files(input_dir: Path, decrypted_dir: Path) -> list[tuple[Path, str | None]]:
    files = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".xlsx" and "微信" in path.name:
            files.append((path, "wechat"))
        elif path.is_file() and path.suffix.lower() == ".csv" and "支付宝" in path.name:
            files.append((path, "alipay"))
    # PDFs in the original input directory may still be encrypted; only the
    # explicitly provided decrypted directory supplies PDF inputs.
    for path in sorted(decrypted_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".pdf":
            files.append((path, "douyin" if path.name.lower().startswith("dy_") or "抖音" in path.name else None))
    return files


def _source_period(metadata: dict, records: list[dict]) -> tuple[str | None, str | None]:
    period = metadata.get("period") or {}
    start = period.get("start") or metadata.get("period_start")
    end = period.get("end") or metadata.get("period_end")
    return start or min((row["time"] for row in records), default=None), end or max((row["time"] for row in records), default=None)


def apply_decisions(transactions: list[dict], decisions: list[dict]) -> list[dict]:
    """Apply explicit user classifications only after every selector is unique.

    A selector must match all its fields on the same canonical row or the same
    evidence row. Multiple matching evidence rows still identify one canonical
    transaction. Validation is atomic, so a later conflict changes no rows.
    """
    if not isinstance(decisions, list):
        raise ValueError("Decisions file must contain a JSON array")
    planned = []
    kinds_by_id = {}
    for index, decision in enumerate(decisions, 1):
        if not isinstance(decision, dict) or set(decision) - {"match", "kind", "category", "reason"}:
            raise ValueError(f"Decision {index}: expected match, kind, reason and optional category")
        selector = decision.get("match")
        required = {"source", "date", "amount", "direction"}
        if not isinstance(selector, dict) or not required <= set(selector) or set(selector) - required - {"counterparty"}:
            raise ValueError(f"Decision {index}: match requires source/date/amount/direction and optional counterparty")
        if selector["source"] not in SOURCE_LABELS:
            raise ValueError(f"Decision {index}: unrecognized source")
        if not isinstance(selector["date"], str) or len(selector["date"]) != 10:
            raise ValueError(f"Decision {index}: date must be YYYY-MM-DD")
        date.fromisoformat(selector["date"])
        if type(selector["amount"]) is not int or selector["amount"] < 0:
            raise ValueError(f"Decision {index}: amount must be nonnegative integer cents")
        if selector["direction"] not in {"in", "out", "neutral"}:
            raise ValueError(f"Decision {index}: direction must be in, out or neutral")
        if "counterparty" in selector and not isinstance(selector["counterparty"], str):
            raise ValueError(f"Decision {index}: counterparty must be a string")
        if decision.get("kind") not in KINDS:
            raise ValueError(f"Decision {index}: invalid economic kind")
        if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError(f"Decision {index}: a nonempty user-confirmed reason is required")
        if "category" in decision and (not isinstance(decision["category"], str) or not decision["category"].strip()):
            raise ValueError(f"Decision {index}: category must be a nonempty string")

        def matches(row: dict) -> bool:
            return (row.get("source") == selector["source"]
                    and str(row.get("time", ""))[:10] == selector["date"]
                    and row.get("amount") == selector["amount"]
                    and row.get("direction") == selector["direction"]
                    and ("counterparty" not in selector or row.get("counterparty") == selector["counterparty"]))

        candidates = [row for row in transactions if any(matches(evidence) for evidence in [row, *row.get("evidence", [])])]
        if len(candidates) != 1:
            raise ValueError(f"Decision {index}: selector matched {len(candidates)} canonical transactions; exactly one is required")
        row = candidates[0]
        previous_kind = kinds_by_id.get(row["id"])
        if previous_kind is not None and previous_kind != decision["kind"]:
            raise ValueError(f"Decision {index}: conflicting kinds target canonical transaction {row['id']}")
        kinds_by_id[row["id"]] = decision["kind"]
        planned.append((row, decision, index))
    applied = []
    for row, decision, index in planned:
        before = {field: row.get(field) for field in ("kind", "category", "review", "reason")}
        row.update(kind=decision["kind"], review=False, review_reason="",
                   reason="用户确认：" + decision["reason"].strip(), confidence="high", user_confirmed=True)
        if "category" in decision:
            row["category"] = decision["category"].strip()
        applied.append({"decision_index": index, "selector": dict(decision["match"]),
                        "record_id": row["id"], "kind": row["kind"], "category": row.get("category"),
                        "reason": row["reason"], "before": before})
    return applied


def build_data(input_dir: Path, decrypted_dir: Path, period_end: str,
               owner_names: tuple[str, ...] = (), decisions: list[dict] | None = None,
               person_aliases: dict[str, str] | None = None) -> dict:
    end_date = date.fromisoformat(period_end)
    period_start = f"{end_date.year}-01-01"
    all_records, errors, file_audits, notes = [], [], [], []
    source_periods = defaultdict(list)
    source_files = defaultdict(list)
    seen_contents = set()
    if not input_dir.is_dir() or not decrypted_dir.is_dir():
        raise ValueError("--input and --decrypted-dir must both be existing directories")
    for path, source_hint in _input_files(input_dir, decrypted_dir):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_contents:
                notes.append(f"内容完全相同的重复文件已跳过：{path.name}")
                continue
            seen_contents.add(digest)
            parser = {"wechat": parse_wechat, "alipay": parse_alipay, "douyin": parse_douyin}.get(source_hint, parse_bank_pdf)
            records, metadata = parser(path)
            source = metadata.get("source") or (records[0]["source"] if records else source_hint)
            if source not in SOURCE_LABELS:
                raise ValueError(f"Unknown source returned by parser: {source}")
            if metadata.get("totals_match") is False or metadata.get("printed_totals_match") is False:
                raise ValueError(f"Statement totals failed reconciliation: {metadata.get('mismatches', [])}")
            for record in records:
                record["id"] = f"{record['id']}@{digest[:12]}"
                record.setdefault("currency", "CNY")
                record["raw"]["source_file"] = path.name
            source_periods[source].append(_source_period(metadata, records))
            source_files[source].append(path.name)
            file_audits.append({"file": path.name, "sha256": digest, "source": source, "metadata": metadata})
            notes.extend(metadata.get("notes", []))
            if metadata.get("audit_note"):
                notes.append(f"{SOURCE_LABELS[source]}：{metadata['audit_note']}")
            all_records.extend(records)
        except Exception as exc:
            # A failed file remains explicit in the deliverable and exit code;
            # successful independent sources can still be reviewed offline.
            errors.append({"file": path.name, "source": source_hint,
                           "type": type(exc).__name__, "message": str(exc)})
    in_period = [row for row in all_records if period_start <= row["time"][:10] <= period_end]
    outside_period = [row for row in all_records if not period_start <= row["time"][:10] <= period_end]
    foreign = [row for row in in_period if row["currency"] != "CNY"]
    domestic = [row for row in in_period if row["currency"] == "CNY"]
    transactions, matches = reconcile(domestic, owner_names=owner_names)
    foreign_cash_audit = annotate_foreign_cash(transactions, foreign)
    applied_decisions = apply_decisions(transactions, decisions if decisions is not None else [])
    pending_cash = [row for row in transactions if row.get("foreign_cash") and row["kind"] == "review"]
    confirmed_cash = [row for row in transactions if row.get("foreign_cash") and row["kind"] == "expense"]
    if pending_cash:
        notes.extend(foreign_cash_audit["notes"])
    if confirmed_cash:
        notes.append(f"用户已确认外币现金用于消费，按实际人民币购汇成本共{sum(row['amount'] for row in confirmed_cash) / 100:.2f}元计入支出；外币提钞本身不再重复计入。")
    self_transfer_audit = audit_self_transfers(transactions)
    matches.extend(self_transfer_audit["matches"])
    self_summary = self_transfer_audit["summary"]
    notes.append(f"已用双向对手账号核对{self_summary['pair_count']}组本人银行互转，共{self_summary['amount_cents'] / 100:.2f}元；双方记账均为内部流转，不计收入或花费。")
    douyin_audit = annotate_douyin_chains(domestic)
    spending_audit = annotate_spending_categories(transactions)
    entities = annotate_entities(transactions, person_aliases=person_aliases)
    monthly = monthly_totals(transactions)
    zero_fields = {"income", "expense", "refund", "reimbursement", "internal", "excluded", "review_in",
                   "review_out", "review_neutral", "review_count", "net_expense", "balance"}
    zero_fields.update(key for values in monthly.values() for key in values)
    for month in range(1, end_date.month + 1):
        monthly.setdefault(f"{end_date.year}-{month:02}", {key: 0 for key in sorted(zero_fields)})
    monthly = dict(sorted(monthly.items()))
    sources = []
    missing_sources = []
    coverage_gaps = []
    for source, label in SOURCE_LABELS.items():
        rows = [row for row in in_period if row["source"] == source]
        cny_rows = [row for row in rows if row["currency"] == "CNY"]
        periods = source_periods.get(source, [])
        starts = [start for start, _ in periods if start]
        ends = [end for _, end in periods if end]
        start, end = min(starts, default=None), max(ends, default=None)
        present = source in source_files
        status = "已读取" if present else "缺失 / 未成功读取"
        if not present:
            missing_sources.append(label)
        elif not rows:
            status = "已读取 / 本期无记录"
        if present and end and end[:10] < period_end:
            coverage_gaps.append({"source": source, "end": end[:10], "requested_end": period_end})
            notes.append(f"{label}的导出范围截至{end[:10]}，之后至{period_end}未覆盖，不能据此认定无交易。")
        sources.append({
            "id": source, "label": label, "count": len(rows),
            "income": sum(row["amount"] for row in cny_rows if row["direction"] == "in"),
            "expense": sum(row["amount"] for row in cny_rows if row["direction"] == "out"),
            "start": start, "end": end, "status": status,
            "files": source_files.get(source, []), "currency": "CNY", "foreign_count": len(rows) - len(cny_rows),
        })
    notes.extend([
        "支出用途按订单摘要、明确商户及来源分类自动识别；不明用途保留其他待分。已关联退款沿用原单用途，未关联的AA回款不猜测原消费类别。",
        "人物及支付方式按去重后的交易统计，一笔交易可能同时出现在商户、代付人和亲情卡下；各对象金额不能再相加。人物全年汇总独立于总览月份筛选。",
        "金额统一以整数分运算。来源收支展示原始流水毛额，最终月度表使用跨来源去重后的经济分类。",
        "商品支出包含亲情卡及他人代付；退款按实际到账月冲减支出。本人账户互转、投资本金和还款不重复计消费。",
        "明确发放的利息及余额宝收益计入收入；基金赎回缺少完整持仓成本，不将本金当作收入，未确认的投资损益不作推算。",
        "AA及报销回款按现有证据暂作支出抵减；带待复核标记的项目可在前端逐笔调整，待确认类型暂不计入收入支出。",
        "浏览器内的审核修正仅保存在当前浏览器；请导出审核记录留存，原始文件和命令行汇总不会被改写。",
        "转至本人其他账户的款项不记花费；未提供完整流水的接收账户，其后续消费无法从这些转账记录补全。",
    ])
    supplied_accounts = {(row.get("bank_name"), row.get("account_tail")) for row in domestic if row["source"].startswith("bank_")}
    referenced_accounts = set()
    for row in domestic:
        referenced_accounts.update(re.findall(r"([\u4e00-\u9fff]{2,8}银行)[^（(]*[（(](\d{4})[）)]", row.get("payment_method", "")))
    uncovered_accounts = sorted(referenced_accounts - supplied_accounts)
    for bank_name, tail in uncovered_accounts:
        notes.append(f"付款记录涉及{bank_name}尾号{tail}，未提供该账户完整银行流水；仅计入本次App中可见交易，账户其他独立收支未覆盖。")
    if foreign:
        currencies = ", ".join(f"{currency} {count}笔" for currency, count in sorted(Counter(row["currency"] for row in foreign).items()))
        notes.append(f"外币流水已单列（{currencies}），未自动折算、未混入人民币月度收支；本报告不代表折算后全币种消费总额。")
    if outside_period:
        notes.append(f"有{len(outside_period)}笔不在{period_start}至{period_end}内，未计入本期汇总。")
    if douyin_audit["summary"]["original_count"]:
        notes.append(douyin_audit["summary"]["interpretation"])
    if errors:
        notes.append(f"有{len(errors)}个文件解析失败；错误详情列于errors及audit.json，当前结果不完整。")
    if applied_decisions:
        notes.append(f"已应用{len(applied_decisions)}项用户明确确认的分类；匹配条件和修改前后信息保存在audit.applied_decisions。")
    audit = {
        "spending_categories": spending_audit,
        "person_aliases": dict(person_aliases or {}),
        "self_transfers": self_transfer_audit,
        "raw_count": len(in_period), "cny_raw_count": len(domestic),
        "canonical_count": len(transactions), "duplicate_count": len(domestic) - len(transactions),
        "internal_count": sum(row["kind"] == "internal" for row in transactions),
        "review_count": sum(row["kind"] == "review" or row.get("review", False) for row in transactions),
        "foreign_count": len(foreign), "outside_period_count": len(outside_period),
        "files": file_audits, "coverage_gaps": coverage_gaps, "douyin_chains": douyin_audit,
        "foreign_cash": foreign_cash_audit,
        "uncovered_accounts": [{"bank_name": bank, "account_tail": tail} for bank, tail in uncovered_accounts],
        "decision_count": len(applied_decisions), "applied_decisions": applied_decisions,
        "errors": errors,
    }
    return {
        "meta": {
            "title": f"{end_date.year}年个人收支", "period_start": period_start, "period_end": period_end,
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "currency": "CNY", "amount_unit": "cents", "fingerprint": fingerprint(domestic),
            "notes": list(dict.fromkeys(notes)), "incomplete": bool(missing_sources or errors or coverage_gaps or uncovered_accounts),
            "missing_sources": missing_sources,
        },
        "sources": sources, "transactions": transactions, "matches": matches, "monthly": monthly,
        "entities": entities, "spending_categories": spending_audit["categories"],
        "foreign_transactions": foreign, "outside_period_transactions": outside_period,
        "audit": audit, "errors": errors,
    }


def _private_write(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            descriptor = -1
            output.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_outputs(data: dict, output_dir: Path) -> None:
    previous_umask = os.umask(0o077)
    try:
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        output_dir.chmod(0o700)
        _private_write(output_dir / "finance-data.json", json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
        _private_write(output_dir / "index.html", render_dashboard(data))
        _private_write(output_dir / "audit.json", json.dumps(data["audit"], ensure_ascii=False, indent=2, allow_nan=False))
        table = StringIO(newline="")
        writer = csv.writer(table)
        columns = [("收入（元）", "income"), ("毛支出（元）", "expense"), ("退款（元）", "refund"),
                   ("AA及报销回款（元）", "reimbursement"), ("净花费（元）", "net_expense"), ("收入减净支出（元）", "balance"),
                   ("待确认收入（元）", "review_in"), ("待确认支出（元）", "review_out"), ("待确认中性（元）", "review_neutral")]
        writer.writerow(["月份"] + [label for label, _ in columns] + ["待复核笔数"])
        for month, values in data["monthly"].items():
            writer.writerow([month] + [format(Decimal(values.get(key, 0)) / 100, ".2f") for _, key in columns] + [values.get("review_count", 0)])
        _private_write(output_dir / "monthly.csv", "\ufeff" + table.getvalue())
    finally:
        os.umask(previous_umask)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing original WeChat XLSX and Alipay CSV")
    parser.add_argument("--decrypted-dir", required=True, type=Path, help="Directory containing already decrypted bank and Douyin PDFs")
    parser.add_argument("--output", required=True, type=Path, help="Destination for the four private offline deliverables")
    parser.add_argument("--owner-name", action="append", default=[], help="Exact account-holder name; repeat for known aliases")
    parser.add_argument("--period-end", default="2026-10-22", help="Inclusive ISO end date (default: 2026-10-22)")
    parser.add_argument("--decisions", type=Path, help="Private JSON file containing explicit user classification decisions")
    parser.add_argument("--person-aliases", type=Path, help="Private JSON object mapping confirmed person aliases to canonical names")
    args = parser.parse_args(argv)
    try:
        decisions = json.loads(args.decisions.read_text(encoding="utf-8-sig")) if args.decisions else None
        person_aliases = json.loads(args.person_aliases.read_text(encoding="utf-8-sig")) if args.person_aliases else None
        data = build_data(args.input, args.decrypted_dir, args.period_end, tuple(args.owner_name), decisions, person_aliases)
        write_outputs(data, args.output)
    except Exception as exc:
        print(f"构建失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    summary = {"output": str(args.output.resolve()), "raw_count": data["audit"]["raw_count"],
               "canonical_count": data["audit"]["canonical_count"], "incomplete": data["meta"]["incomplete"],
               "missing_sources": data["meta"]["missing_sources"], "error_count": len(data["errors"])}
    print(json.dumps(summary, ensure_ascii=False))
    for error in data["errors"]:
        print(f"解析失败：{error['file']}：{error['type']}: {error['message']}", file=sys.stderr)
    return 2 if data["errors"] or data["meta"]["missing_sources"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
