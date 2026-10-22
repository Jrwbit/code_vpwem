"""Synthetic accounting regressions; no private account data belongs here."""
from copy import deepcopy

import pytest

from tools.personal_finance.reconcile import find_refunds, monthly_totals, reconcile


def record(identifier, source="alipay", amount=10_000, direction="out", **fields):
    result = {
        "id": identifier,
        "source": source,
        "source_label": source,
        "row": 1,
        "time": "2026-01-10 12:00:00",
        "direction": direction,
        "amount": amount,
        "counterparty": "测试商户甲",
        "description": "测试商品",
        "transaction_type": "日用百货",
        "payment_method": "测试银行储蓄卡(1234)",
        "status": "交易成功",
        "transaction_id": identifier,
        "merchant_id": "",
    }
    result.update(fields)
    return result


def bank_record(identifier="bank-1", **fields):
    values = {
        "counterparty": "支付宝",
        "description": "支付宝消费",
        "bank_name": "测试银行",
        "account_tail": "1234",
        "time_precision": "second",
        "transaction_type": "消费",
    }
    values.update(fields)
    return record(identifier, source="bank_a", **values)


def by_id(records):
    return {item["id"]: item for item in records}


def test_closed_order_with_completed_refund_retains_original_gross_expense():
    purchase = record("purchase", transaction_id="order-1", status="交易关闭")
    refund = record("refund", transaction_id="order-1_refund", status="退款成功",
                    direction="neutral", amount=2_500, time="2026-02-03 10:00:00")
    abandoned = record("abandoned", status="交易关闭", amount=9_999)
    transactions, matches = reconcile([purchase, refund, abandoned])
    items = by_id(transactions)
    assert items["purchase"]["kind"] == "expense"
    assert items["refund"]["kind"] == "refund"
    assert items["abandoned"]["kind"] == "excluded"
    totals = monthly_totals(transactions)
    assert totals["2026-01"]["net_expense"] == 10_000
    assert totals["2026-02"]["net_expense"] == -2_500
    assert sum(m["net_expense"] for m in totals.values()) == 7_500
    assert [m["record_ids"] for m in matches if m["type"] == "refund_link"] == [["purchase", "refund"]]


def test_multiple_partial_refund_events_are_not_collapsed():
    records = [record("buy", transaction_id="parent", status="交易关闭"),
               record("refund-1", transaction_id="parent_r1", status="退款成功",
                      direction="neutral", amount=2_500),
               record("refund-2", transaction_id="parent_r2", status="退款成功",
                      direction="neutral", amount=2_500)]
    transactions, _ = reconcile(records)
    assert len(transactions) == 3
    assert monthly_totals(transactions)["2026-01"]["net_expense"] == 5_000


@pytest.mark.parametrize("method", ["亲情卡", "他人代付"])
def test_user_consumption_includes_other_person_funded_neutral_purchase(method):
    transactions, _ = reconcile([
        record("buy", direction="neutral", payment_method=method),
        record("refund", transaction_id="buy_r", status="退款成功", direction="neutral",
               amount=3_000, payment_method=method),
    ])
    items = by_id(transactions)
    assert items["buy"]["kind"] == "expense"
    assert items["buy"]["funded_by_other"] is True
    assert items["refund"]["kind"] == "refund"
    assert items["refund"]["funded_by_other"] is True
    assert monthly_totals(transactions)["2026-01"]["net_expense"] == 7_000


def test_family_payment_failed_order_does_not_become_consumption():
    transactions, _ = reconcile([record("failed", payment_method="亲情卡", direction="neutral",
                                          status="支付失败")])
    assert transactions[0]["kind"] == "excluded"
    assert monthly_totals(transactions)["2026-01"]["net_expense"] == 0


def test_group_receipt_is_provisional_reimbursement_not_income():
    receipt = record("aa", source="wechat", direction="in", amount=3_000,
                     transaction_type="群收款", payment_method="零钱")
    transactions, _ = reconcile([record("meal"), receipt])
    assert by_id(transactions)["aa"]["review"] is True
    assert by_id(transactions)["aa"]["kind"] == "reimbursement"
    totals = monthly_totals(transactions)["2026-01"]
    assert totals["income"] == 0
    assert totals["reimbursement"] == 3_000
    assert totals["net_expense"] == 7_000
    assert totals["review_count"] == 1


def test_explicit_self_transfer_is_internal_but_similar_name_is_not():
    transactions, _ = reconcile([
        record("self", counterparty="测试本人", transaction_type="转账"),
        record("other", counterparty="测试本人乙", transaction_type="转账"),
    ], owner_names=("测试本人",))
    assert by_id(transactions)["self"]["kind"] == "internal"
    assert by_id(transactions)["other"]["kind"] != "internal"


def test_same_amount_and_minute_to_different_people_keeps_distinct_transactions():
    transactions, matches = reconcile([
        record("person-a", source="wechat", counterparty="测试张甲", transaction_type="转账"),
        record("person-b", source="wechat", counterparty="测试李乙", transaction_type="转账"),
    ])
    assert len(transactions) == 2
    assert not matches
    assert all(t["review"] for t in transactions)


def test_unique_bank_app_match_preserves_both_evidence_records_and_counts_once():
    raw = [record("purchase"), bank_record()]
    original = deepcopy(raw)
    transactions, matches = reconcile(raw)
    assert raw == original, "Reconciliation must not mutate parsed source evidence"
    assert len(transactions) == 1
    assert set(transactions[0]["sources"]) == {"alipay", "bank_a"}
    assert {e["id"] for e in transactions[0]["evidence"]} == {"purchase", "bank-1"}
    assert monthly_totals(transactions)["2026-01"]["expense"] == 10_000
    assert len([m for m in matches if m["type"] == "duplicate"]) == 1


@pytest.mark.parametrize("bank_fields", [
    {"description": "财付通微信消费", "counterparty": "财付通"},
    {"account_tail": "9876"},
    {"time": "2026-01-10 12:05:01"},
    {"time": "2026-01-11 12:00:00"},
    {"direction": "in"},
    {"amount": 10_001},
])
def test_bank_match_requires_channel_account_time_direction_and_exact_cents(bank_fields):
    transactions, matches = reconcile([record("purchase"), bank_record(**bank_fields)])
    assert len(transactions) == 2
    assert not any(m["type"] == "duplicate" for m in matches)


def test_day_precision_bank_can_match_unique_same_day_app_purchase():
    transactions, _ = reconcile([
        record("purchase", time="2026-01-10 23:25:00"),
        bank_record(time="2026-01-10 00:00:00", time_precision="day"),
    ])
    assert len(transactions) == 1


def test_one_bank_two_same_amount_app_candidates_keeps_bank_out_of_total_pending_review():
    transactions, matches = reconcile([
        record("purchase-a", counterparty="测试商户甲"),
        record("purchase-b", counterparty="测试商户乙", time="2026-01-10 12:00:00"),
        bank_record(),
    ])
    assert len(transactions) == 3
    assert not any(m["type"] == "duplicate" for m in matches)
    assert by_id(transactions)["bank-1"]["kind"] == "review"
    assert set(by_id(transactions)["bank-1"]["candidate_ids"]) == {"purchase-a", "purchase-b"}
    totals = monthly_totals(transactions)["2026-01"]
    assert totals["expense"] == 20_000
    assert totals["review_out"] == 10_000


def test_two_bank_rows_one_app_candidate_is_not_greedily_matched():
    transactions, matches = reconcile([
        record("purchase"), bank_record("bank-1"),
        bank_record("bank-2", time="2026-01-10 12:01:00"),
    ])
    assert len(transactions) == 3
    assert not any(m["type"] == "duplicate" for m in matches)
    assert by_id(transactions)["bank-1"]["kind"] == "review"
    assert by_id(transactions)["bank-2"]["kind"] == "review"
    totals = monthly_totals(transactions)["2026-01"]
    assert totals["expense"] == 10_000
    assert totals["review_out"] == 20_000


def test_neutral_unclassified_amount_remains_visible_in_pending_bucket():
    transactions, _ = reconcile([record("unknown", direction="neutral", amount=7_700,
                                          payment_method="", transaction_type="其他")])
    assert transactions[0]["kind"] == "review"
    totals = monthly_totals(transactions)["2026-01"]
    assert totals["review_neutral"] == 7_700
    assert totals["review_count"] == 1
    assert totals["net_expense"] == 0
    assert totals["income"] == 0


def test_same_export_event_repeated_counts_once_but_keeps_evidence():
    transactions, matches = reconcile([
        record("copy-one", transaction_id="stable-order"),
        record("copy-two", transaction_id="stable-order", row=12),
    ])
    assert len(transactions) == 1
    assert len(transactions[0]["evidence"]) == 2
    assert monthly_totals(transactions)["2026-01"]["expense"] == 10_000
    assert matches[0]["type"] == "duplicate"


def test_duplicate_raw_record_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        reconcile([record("same"), record("same")])


def test_nonunique_refund_parent_order_key_is_not_arbitrarily_linked():
    """Installment suffixes may identify distinct settled parent events."""
    purchase_a = record("installment-a", transaction_id="parent_part1", amount=6_000)
    purchase_b = record("installment-b", transaction_id="parent_part2", amount=4_000)
    refund = record("refund", transaction_id="parent_refund", status="退款成功",
                    direction="neutral", amount=1_000)
    assert find_refunds([purchase_a, purchase_b, refund]) == {}
    assert find_refunds([purchase_b, purchase_a, refund]) == {}


def test_known_different_banks_with_same_account_tail_are_not_merged():
    transactions, matches = reconcile([
        record("purchase", payment_method="另一银行储蓄卡(1234)"),
        bank_record(bank_name="测试银行", account_tail="1234"),
    ])
    assert len(transactions) == 2
    assert not any(m["type"] == "duplicate" for m in matches)


def test_explicit_conflicting_counterparty_prevents_unique_amount_match():
    transactions, matches = reconcile([
        record("purchase", counterparty="测试商户甲"),
        bank_record(counterparty="测试商户乙", description="支付宝消费"),
    ])
    assert len(transactions) == 2
    assert not any(m["type"] == "duplicate" for m in matches)


def test_repeated_closed_purchase_with_refund_keeps_gross_purchase_after_dedup():
    """A refund must activate the retained original, not only the discarded copy."""
    records = [
        record("copy-a", transaction_id="shared-order", status="交易关闭"),
        record("copy-b", transaction_id="shared-order", status="交易关闭", row=12),
        record("refund", transaction_id="shared-order_refund", status="退款成功",
               direction="neutral", amount=2_500),
    ]
    transactions, _ = reconcile(records)
    assert len(transactions) == 2
    totals = monthly_totals(transactions)["2026-01"]
    assert totals["expense"] == 10_000
    assert totals["refund"] == 2_500
    assert totals["net_expense"] == 7_500
