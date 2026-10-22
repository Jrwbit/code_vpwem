"""Audit Douyin monthly-payment authorizations and subsequent confirmations.

The four-digit routing suffix is not an order ID. Links below remain evidence
candidates and are only corroborated when the complete per-order amount identity
holds: original payment = ONP outgoing confirmations + ONP incoming reversals.
INP refunds happen after confirmation and reduce the retained purchase amount.
"""

from __future__ import annotations


def annotate_douyin_chains(records: list[dict]) -> dict:
    """Return candidate chains and exact arithmetic; do not mutate input rows."""
    douyin = [record for record in records if record["source"] == "douyin"]
    originals = [record for record in douyin if record["direction"] == "out"
                 and record["payment_method"] == "抖音月付"
                 and not record["merchant_id"].startswith("ONP")]
    grouped = {record["id"]: [] for record in originals}
    ambiguous = []
    unmatched_onp = []
    for record in douyin:
        prefix = record["merchant_id"][:3]
        if prefix not in {"ONP", "INP"}:
            continue
        candidates = [original for original in originals
                      if original["transaction_id"][-4:] == record["transaction_id"][-4:]
                      and original["counterparty"] == record["counterparty"]
                      and original["time"] <= record["time"]
                      and record["amount"] <= original["amount"]]
        if len(candidates) == 1:
            grouped[candidates[0]["id"]].append(record)
        elif len(candidates) > 1:
            ambiguous.append({"record_id": record["id"], "candidate_original_ids": [row["id"] for row in candidates]})
        elif prefix == "ONP":
            unmatched_onp.append(record["id"])
    chains = []
    for original in originals:
        events = sorted(grouped[original["id"]], key=lambda record: (record["time"], record["id"]))
        onp_out = [record for record in events if record["merchant_id"].startswith("ONP") and record["direction"] == "out"]
        onp_in = [record for record in events if record["merchant_id"].startswith("ONP") and record["direction"] == "in"]
        inp_in = [record for record in events if record["merchant_id"].startswith("INP") and record["direction"] == "in"]
        out_amount = sum(record["amount"] for record in onp_out)
        onp_refund_amount = sum(record["amount"] for record in onp_in)
        inp_refund_amount = sum(record["amount"] for record in inp_in)
        difference = original["amount"] - out_amount - onp_refund_amount
        refunds_within_original = onp_refund_amount + inp_refund_amount <= original["amount"]
        evidence_balanced = difference == 0 and refunds_within_original
        chains.append({
            "original_id": original["id"], "original_row": original["row"],
            "original_time": original["time"], "original_amount": original["amount"],
            "routing_suffix": original["transaction_id"][-4:],
            "onp_outgoing_ids": [record["id"] for record in onp_out],
            "onp_outgoing_rows": [record["row"] for record in onp_out],
            "onp_outgoing_amount": out_amount,
            "onp_refund_ids": [record["id"] for record in onp_in],
            "onp_refund_rows": [record["row"] for record in onp_in],
            "onp_refund_amount": onp_refund_amount,
            "inp_refund_ids": [record["id"] for record in inp_in],
            "inp_refund_rows": [record["row"] for record in inp_in],
            "inp_refund_amount": inp_refund_amount,
            "identity_difference": difference,
            "refunds_within_original": refunds_within_original,
            "balanced": evidence_balanced,
            "candidate_net_expense": original["amount"] - onp_refund_amount - inp_refund_amount,
            "evidence": "月付原单与后续记录的交易单号末4位、交易对方及先后时间一致；核验原单金额等于ONP支出加ONP退款。末4位不是完整订单外键。",
        })
    balanced = [chain for chain in chains if chain["balanced"]]
    duplicate_candidates = [identifier for chain in balanced for identifier in chain["onp_outgoing_ids"]]
    gross_expense = sum(record["amount"] for record in douyin if record["direction"] == "out")
    gross_refund = sum(record["amount"] for record in douyin if record["direction"] == "in" and record["transaction_type"] == "退款")
    candidate_duplicate_amount = sum(chain["onp_outgoing_amount"] for chain in balanced)
    return {
        "chains": chains,
        "ambiguous_events": ambiguous,
        "unmatched_onp_ids": unmatched_onp,
        "onp_expense_duplicate_candidate_ids": duplicate_candidates,
        "summary": {
            "original_count": len(originals),
            "original_amount": sum(record["amount"] for record in originals),
            "balanced_chain_count": len(balanced),
            "all_chains_balanced": bool(chains) and len(chains) == len(balanced) and not ambiguous and not unmatched_onp,
            "onp_outgoing_count": sum(len(chain["onp_outgoing_ids"]) for chain in chains),
            "onp_outgoing_amount": sum(chain["onp_outgoing_amount"] for chain in chains),
            "onp_refund_count": sum(len(chain["onp_refund_ids"]) for chain in chains),
            "onp_refund_amount": sum(chain["onp_refund_amount"] for chain in chains),
            "inp_refund_count": sum(len(chain["inp_refund_ids"]) for chain in chains),
            "inp_refund_amount": sum(chain["inp_refund_amount"] for chain in chains),
            "mixed_refund_kinds_chain_count": sum(bool(chain["onp_refund_ids"] and chain["inp_refund_ids"]) for chain in chains),
            "over_refunded_chain_count": sum(not chain["refunds_within_original"] for chain in chains),
            "raw_expense_amount": gross_expense,
            "raw_refund_amount": gross_refund,
            "candidate_duplicate_expense_amount": candidate_duplicate_amount,
            "candidate_expense_after_dedup": gross_expense - candidate_duplicate_amount,
            "candidate_net_expense_after_dedup": gross_expense - candidate_duplicate_amount - gross_refund,
            "interpretation": "ONP支出符合原月付订单的后续核销/确认模式，建议不重复计消费；ONP退款与INP退款均按实际退款日保留。本结论来自整组金额闭环，非完整订单ID关联。",
        },
    }
