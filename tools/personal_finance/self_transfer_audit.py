"""Read-only corroboration of both bank postings for one personal transfer."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import re


def _counterparty_accounts(record: dict) -> set[str]:
    raw = record.get("raw", {})
    text = " ".join(str(raw.get(field, "")) for field in (
        "counterparty_account", "counterparty_info", "counterparty_account_bank",
    ))
    return set(re.findall(r"\d+", text))


def _location(record: dict) -> str:
    label = record.get("source_label", record["source"])
    raw = record.get("raw", {})
    if raw.get("page") is not None and raw.get("page_row") is not None:
        row = f"第{raw['page']}页第{raw['page_row']}行"
    else:
        row = f"第{record.get('row', '未注明')}行"
    return f"{label} {record['time'][:10]} {row}"


def audit_self_transfers(transactions: list[dict]) -> dict:
    """Return paired transfer evidence without changing transactions/evidence.

    Only CNY bank rows already classified as ``internal / 本人转账`` qualify.
    Pairing requires equal positive amounts, opposite directions, different
    source accounts, posting times within 24 hours, reciprocal counterparty
    account numbers, and a unique candidate in both directions. An unmatched
    leg stays an internal transfer; no new income or expense is inferred.
    """
    rows = [record for record in transactions
            if record.get("source", "").startswith("bank_")
            and record.get("currency", "CNY") == "CNY"
            and record.get("kind") == "internal" and record.get("category") == "本人转账"]
    outgoing = [record for record in rows if record.get("direction") == "out" and record["amount"] > 0]
    incoming = [record for record in rows if record.get("direction") == "in" and record["amount"] > 0]
    candidates, reverse = {}, defaultdict(list)
    for out in outgoing:
        account = str(out.get("raw", {}).get("account", ""))
        out_targets = _counterparty_accounts(out)
        choices = []
        if account:
            for inc in incoming:
                other_account = str(inc.get("raw", {}).get("account", ""))
                if (not other_account or other_account == account
                        or out["amount"] != inc["amount"]
                        or other_account not in out_targets
                        or account not in _counterparty_accounts(inc)):
                    continue
                difference = abs((datetime.fromisoformat(out["time"]) - datetime.fromisoformat(inc["time"])).total_seconds())
                if difference <= 24 * 60 * 60:
                    choices.append(inc)
                    reverse[inc["id"]].append(out)
        candidates[out["id"]] = choices

    matches, paired_ids = [], set()
    for out in outgoing:
        choices = candidates[out["id"]]
        if len(choices) != 1 or len(reverse[choices[0]["id"]]) != 1:
            continue
        inc = choices[0]
        matches.append({
            "id": f"internal-transfer:{out['id']}:{inc['id']}",
            "type": "internal_transfer", "record_ids": [out["id"], inc["id"]],
            "amount": out["amount"],
            "reason": f"{_location(out)} 转出，与 {_location(inc)} 转入等额且相隔不超过1天；双方对手账号互相指向，唯一对应。同一次本人转账的两条记账记录，均保留内部流转。",
        })
        paired_ids.update((out["id"], inc["id"]))

    single = []
    for record in rows:
        if record["id"] in paired_ids:
            continue
        choices = candidates.get(record["id"], reverse.get(record["id"], []))
        single.append({
            "record_id": record["id"], "source": record["source"], "time": record["time"],
            "row": record.get("row"), "direction": record["direction"], "amount": record["amount"],
            "candidate_ids": [choice["id"] for choice in choices],
            "reason": "未找到双方账号互相指向的唯一转账对；保留原有本人内部流转分类。",
        })
    amount = sum(match["amount"] for match in matches)
    return {
        "matches": matches,
        "summary": {
            "currency": "CNY", "amount_unit": "cents", "self_leg_count": len(rows),
            "pair_count": len(matches), "amount_cents": amount,
            "matched_leg_count": len(paired_ids), "matched_leg_amount_cents": 2 * amount,
            "single_self_leg_count": len(single),
        },
        "single_self_legs": single,
    }
