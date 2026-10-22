"""Identify newly purchased foreign currency withdrawn as unverified cash.

This is an evidence link, not an exchange-rate estimator. The CNY purchase
amount is retained exactly; foreign minor units are never added to CNY totals.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import re


def _summary(record: dict) -> str:
    return record.get("raw", {}).get("summary", record.get("description", "").split("；", 1)[0])


def _account(record: dict) -> tuple[str, str] | None:
    source = record.get("source", "")
    account = record.get("raw", {}).get("account", "")
    return (source, account) if source.startswith("bank_") and account else None


def _time(record: dict) -> datetime:
    return datetime.fromisoformat(record["time"])


def annotate_foreign_cash(transactions: list[dict], foreign_records: list[dict]) -> dict:
    """Mark an exactly linked CNY purchase as ``review`` in place.

    A CNY debit and foreign credit must both be purchases (购汇), belong to the
    same bank account, and be uniquely paired within the same minute. That
    foreign credit must be followed by a unique, equal-amount cash withdrawal
    within 30 minutes, with no intervening entries for that currency/account.
    Available balances must also reconcile. Other foreign entries are untouched.

    Only CNY canonical rows are reclassified. The two foreign records are copied
    into ``evidence`` with their original currency and amount; callers must render
    those evidence amounts using their currency. Run before user decisions so a
    later confirmed expense can override the review classification.
    """
    cny_purchases = [record for record in transactions
                     if record.get("currency", "CNY") == "CNY"
                     and record.get("direction") == "out" and record["amount"] > 0
                     and _summary(record) == "购汇" and _account(record)]
    foreign = [record for record in foreign_records if record.get("currency", "CNY") != "CNY"]
    foreign_purchases = [record for record in foreign if record.get("direction") == "in"
                         and record["amount"] > 0 and _summary(record) == "购汇" and _account(record)]
    candidates = {}
    reverse = defaultdict(list)
    for cny in cny_purchases:
        choices = [record for record in foreign_purchases
                   if _account(record) == _account(cny)
                   and record["time"][:16] == cny["time"][:16]]
        candidates[cny["id"]] = choices
        for record in choices:
            reverse[record["id"]].append(cny)

    matches = []
    by_currency = defaultdict(int)
    for cny in cny_purchases:
        choices = candidates[cny["id"]]
        if len(choices) != 1 or len(reverse[choices[0]["id"]]) != 1:
            continue
        purchase = choices[0]
        ledger = [record for record in foreign if _account(record) == _account(purchase)
                  and record["currency"] == purchase["currency"]]
        cash_choices = [record for record in ledger
                        if record.get("direction") == "out"
                        and re.search(r"取款|支取", _summary(record))
                        and record["amount"] == purchase["amount"]
                        and 0 <= (_time(record) - _time(purchase)).total_seconds() <= 30 * 60]
        if len(cash_choices) != 1:
            continue
        withdrawal = cash_choices[0]
        intervening = [record for record in ledger
                       if record["id"] not in {purchase["id"], withdrawal["id"]}
                       and _time(purchase) <= _time(record) <= _time(withdrawal)]
        if intervening:
            continue
        before = purchase.get("raw", {}).get("balance_cents")
        after = withdrawal.get("raw", {}).get("balance_cents")
        if before is not None and after is not None and before - withdrawal["amount"] != after:
            continue

        match = {
            "cny_record_id": cny["id"], "cny_amount": cny["amount"],
            "foreign_purchase_id": purchase["id"], "foreign_withdrawal_id": withdrawal["id"],
            "foreign_currency": purchase["currency"], "foreign_amount": withdrawal["amount"],
            "foreign_amount_unit": "one_hundredth_of_original_currency",
            "withdrawal_delay_seconds": int((_time(withdrawal) - _time(purchase)).total_seconds()),
            "reason": "两条购汇同分钟且唯一对应；同账户同币种随后30分钟内唯一等额提钞，中间无其他流水。",
        }
        cny.update(
            kind="review", category="外币现金待核", review=True, confidence="medium",
            reason="该笔人民币购汇对应外币现金提取；提钞不能证明已经消费，暂不计入净消费。",
            review_reason="请确认提取的外币现金实际花费；全部已花可将本笔人民币金额确认为支出。",
            foreign_cash=match,
        )
        evidence = cny.setdefault("evidence", [])
        evidence_ids = {record["id"] for record in evidence}
        for record in (purchase, withdrawal):
            if record["id"] not in evidence_ids:
                evidence.append(deepcopy(record))
                evidence_ids.add(record["id"])
        matches.append(match)
        by_currency[purchase["currency"]] += withdrawal["amount"]

    cny_total = sum(match["cny_amount"] for match in matches)
    notes = []
    if matches:
        notes.append(f"{len(matches)}笔人民币购汇共{cny_total / 100:.2f}元与随后外币提钞对应，已列外币现金待核；未将提钞直接视为消费。")
    return {
        "matches": matches, "matched_cash_count": len(matches),
        "matched_cash_cny_cents": cny_total,
        "foreign_cash_by_currency": dict(by_currency),
        "foreign_amount_unit": "one_hundredth_of_original_currency", "notes": notes,
    }
