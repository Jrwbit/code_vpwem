"""Synthetic regressions for delayed postings and multi-leg payment matching."""

from copy import deepcopy

import pytest

from tools.personal_finance.reconcile import monthly_totals, reconcile


def transaction(identifier, source="alipay", amount=10_000, direction="out", **overrides):
    row = {
        "id": identifier, "source": source, "source_label": source, "row": 1,
        "time": "2026-01-10 12:00:00", "amount": amount, "direction": direction,
        "counterparty": "示例商户", "description": "示例商品", "transaction_type": "日用百货",
        "payment_method": "中国银行储蓄卡(1234)", "status": "交易成功",
        "transaction_id": identifier, "merchant_id": "", "raw": {},
    }
    row.update(overrides)
    return row


def bank(identifier, **overrides):
    defaults = {
        "source": "bank_boc", "bank_name": "中国银行", "account_tail": "1234",
        "counterparty": "支付宝", "description": "支付宝快捷支付", "transaction_type": "消费",
        "time_precision": "second", "status": "已入账",
    }
    defaults.update(overrides)
    return transaction(identifier, **defaults)


def douyin(identifier, amount, time, prefix="200", direction="out", method="抖音月付", suffix="4321"):
    return transaction(identifier, source="douyin", amount=amount, direction=direction,
                       time=time, counterparty="抖音生活服务", transaction_type="退款" if direction == "in" else "支付",
                       payment_method=method, status="", transaction_id=f"{identifier}-{suffix}",
                       merchant_id=f"{prefix}{identifier}")


def test_boc_posting_two_hours_after_midnight_keeps_purchase_month():
    purchase = transaction("purchase", time="2026-01-31 23:30:00")
    posting = bank("posting", time="2026-02-01 01:30:00")
    rows, matches = reconcile([purchase, posting])
    assert len(rows) == 1
    assert rows[0]["id"] == "purchase"
    assert {item["id"] for item in rows[0]["evidence"]} == {"purchase", "posting"}
    assert rows[0]["confidence"] == "medium"
    assert monthly_totals(rows)["2026-01"]["expense"] == 10_000
    assert "2026-02" not in monthly_totals(rows)
    assert any(item["type"] == "duplicate" for item in matches)


@pytest.mark.parametrize("posting_time", ["2026-01-10 11:54:59", "2026-01-10 10:00:00"])
def test_boc_posting_over_five_minutes_before_app_does_not_match(posting_time):
    rows, matches = reconcile([transaction("purchase"), bank("posting", time=posting_time)])
    assert len(rows) == 2
    assert not any(item["type"].startswith("duplicate") for item in matches)


def test_another_bank_does_not_inherit_boc_six_hour_delay_window():
    rows, matches = reconcile([
        transaction("purchase", time="2026-01-31 23:30:00"),
        bank("posting", source="bank_cib", time="2026-02-01 01:30:00"),
    ])
    assert len(rows) == 2
    assert not matches


@pytest.mark.parametrize("description", ["快捷支付-美团支付", "银联无卡消费 美团支付"])
def test_explicit_meituan_channel_is_not_mistaken_for_alipay(description):
    rows, matches = reconcile([
        transaction("purchase", counterparty="美团"),
        bank("posting", counterparty="美团", description=description),
    ])
    assert len(rows) == 2
    assert not any(item["type"].startswith("duplicate") for item in matches)


def test_balanced_douyin_monthly_chain_keeps_original_and_refund_only():
    raw = [
        douyin("original", 10_000, "2026-01-01 12:00:00"),
        douyin("confirmation-a", 3_000, "2026-01-02 12:00:00", "ONP"),
        douyin("confirmation-b", 3_000, "2026-01-03 12:00:00", "ONP"),
        douyin("refund", 4_000, "2026-01-04 12:00:00", "ONP", "in", "退款"),
    ]
    before = deepcopy(raw)
    rows, matches = reconcile(raw)
    assert raw == before
    assert {item["id"] for item in rows} == {"original", "refund"}
    original = next(item for item in rows if item["id"] == "original")
    assert original["review"] is True
    assert {item["id"] for item in original["evidence"]} == {"original", "confirmation-a", "confirmation-b"}
    totals = monthly_totals(rows)["2026-01"]
    assert (totals["expense"], totals["refund"], totals["net_expense"]) == (10_000, 4_000, 6_000)
    assert len([item for item in matches if item["type"] == "duplicate"]) == 2


def test_douyin_wallet_card_purchase_and_full_refund_keep_full_amount_once():
    purchase = douyin("purchase", 10_000, "2026-01-10 12:00:00",
                      method="抖音零钱&中国银行借记卡（1234）")
    card_payment = bank("card-payment", amount=6_000, time="2026-01-10 12:00:20",
                        counterparty="抖音支付", description="抖音支付消费")
    refund = douyin("refund", 10_000, "2026-01-12 12:00:00", "INP", "in", "退款")
    card_refund = bank("card-refund", amount=6_000, direction="in", time="2026-01-12 12:00:30",
                       counterparty="抖音支付", description="抖音支付退款")
    rows, matches = reconcile([purchase, card_payment, refund, card_refund])
    assert {item["id"] for item in rows} == {"purchase", "refund"}
    by_id = {item["id"]: item for item in rows}
    assert {item["id"] for item in by_id["purchase"]["evidence"]} == {"purchase", "card-payment"}
    assert {item["id"] for item in by_id["refund"]["evidence"]} == {"refund", "card-refund"}
    totals = monthly_totals(rows)["2026-01"]
    assert (totals["expense"], totals["refund"], totals["net_expense"]) == (10_000, 10_000, 0)
    assert len([item for item in matches if item["type"] == "duplicate"]) == 2


def same_amount_group(app_count, bank_count):
    # No exact second match: every bank must retain all same-amount candidates.
    apps = [transaction(f"app-{index}", direction="in", status="退款成功",
                        time=f"2026-01-10 12:00:{index:02}", transaction_type="退款")
            for index in range(app_count)]
    postings = [bank(f"bank-{index}", direction="in", description="支付宝退款",
                     time=f"2026-01-10 12:02:{index:02}") for index in range(bank_count)]
    return apps + postings


def test_complete_two_by_two_same_month_group_deduplicates_with_review():
    rows, matches = reconcile(same_amount_group(2, 2))
    assert {item["id"] for item in rows} == {"app-0", "app-1"}
    assert all(item["review"] is True for item in rows)
    assert all(item["confidence"] == "medium" for item in rows)
    assert {evidence["id"] for item in rows for evidence in item["evidence"]} == {"app-0", "app-1", "bank-0", "bank-1"}
    assert monthly_totals(rows)["2026-01"]["refund"] == 20_000
    groups = [item for item in matches if item["type"] == "duplicate_group"]
    assert len(groups) == 1
    assert groups[0]["amount"] == 20_000
    assert set(groups[0]["record_ids"]) == {"app-0", "app-1", "bank-0", "bank-1"}


@pytest.mark.parametrize("app_count,bank_count", [(2, 1), (2, 3), (3, 2)])
def test_unequal_candidate_group_counts_are_not_removed(app_count, bank_count):
    rows, matches = reconcile(same_amount_group(app_count, bank_count))
    assert len(rows) == app_count + bank_count
    assert not any(item["type"].startswith("duplicate") for item in matches)
    postings = [item for item in rows if item["source"] == "bank_boc"]
    assert all(item["kind"] == "review" and item["review"] for item in postings)
    assert monthly_totals(rows)["2026-01"]["refund"] == 10_000 * app_count


def test_equal_candidate_group_across_months_is_not_collapsed():
    raw = same_amount_group(2, 2)
    for item in raw:
        item["time"] = "2026-01-31 23:59:00" if item["source"] == "alipay" else "2026-02-01 00:01:00"
    rows, matches = reconcile(raw)
    assert len(rows) == 4
    assert not any(item["type"].startswith("duplicate") for item in matches)
