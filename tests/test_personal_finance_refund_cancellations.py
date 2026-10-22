from copy import deepcopy

import pytest

from tools.personal_finance.refund_cancellations import remove_fully_refunded


def row(identifier, kind="expense", amount=10000, **fields):
    result = {"id": identifier, "kind": kind, "amount": amount, "currency": "CNY",
              "time": "2026-01-02 10:00:00", "source": "alipay", "category": "日用百货",
              "description": "商品订单", "counterparty": "示例商户", "merchant_id": "",
              "evidence": [], "review": True, "review_reason": "原复核依据", "reason": "原统计依据"}
    result.update(fields)
    return result


def test_cross_month_full_refund_resolves_merged_evidence_and_is_idempotent():
    rows = [row("purchase", evidence=[{"id": "old-bank-row", "amount": 10000}]),
            row("refund", "refund", related_id="old-bank-row", time="2026-02-03 10:00:00")]
    before = deepcopy(rows)
    audit = remove_fully_refunded(rows)
    assert audit["summary"] == {"group_count": 1, "record_count": 2, "expense_removed": 10000,
                                "refund_removed": 10000, "cross_month_count": 1}
    assert all(item["kind"] == "excluded" and not item["review"] for item in rows)
    for old, new in zip(before, rows):
        assert [new[key] for key in ("amount", "category", "evidence")] == [old[key] for key in ("amount", "category", "evidence")]
        assert new["removal_original_reason"] == old["reason"]
        assert new["removal_original_review"] is True
        assert new["removal_original_review_reason"] == "原复核依据"
    once = deepcopy(rows)
    assert remove_fully_refunded(rows) == audit
    assert rows == once


def test_multiple_partial_refunds_can_close_one_full_purchase():
    rows = [row("purchase"), row("refund-a", "refund", 3000, related_id="purchase"),
            row("refund-b", "refund", 7000, related_id="purchase", evidence=[{"id": "bank-refund", "amount": 7000}])]
    audit = remove_fully_refunded(rows)
    assert audit["summary"]["group_count"] == 1
    assert audit["summary"]["record_count"] == 3
    assert audit["summary"]["refund_removed"] == 10000


@pytest.mark.parametrize("amount", [9000, 11000])
def test_partial_and_excess_refunds_do_not_remove_any_records(amount):
    rows = [row("purchase"), row("refund", "refund", amount, related_id="purchase")]
    before = deepcopy(rows)
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0
    assert rows == before


def test_matching_amount_without_order_relationship_is_not_enough():
    rows = [row("purchase", description="抖音生活服务", counterparty="抖音生活服务"),
            row("refund", "refund", description="抖音生活服务退款", counterparty="抖音生活服务")]
    before = deepcopy(rows)
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0
    assert rows == before


def test_explicit_refund_link_match_is_used_when_related_id_absent():
    rows = [row("purchase"), row("refund", "refund")]
    matches = [{"type": "refund_link", "record_ids": ["purchase", "refund"]}]
    assert remove_fully_refunded(rows, matches)["summary"]["group_count"] == 1


def test_conflicting_parent_links_leave_all_records_unchanged():
    rows = [row("purchase-a"), row("purchase-b"), row("refund", "refund", related_id="purchase-a")]
    matches = [{"type": "refund_link", "record_ids": ["purchase-b", "refund"]}]
    before = deepcopy(rows)
    audit = remove_fully_refunded(rows, matches)
    assert audit["summary"]["group_count"] == 0
    assert audit["diagnostics"]["conflicting_refund_count"] == 1
    assert rows == before


@pytest.mark.parametrize("parent_fields,refund_fields", [
    ({"kind": "internal"}, {}), ({}, {"currency": "USD"}),
    ({"time": "2026-02-01 00:00:00"}, {"time": "2026-01-01 00:00:00"}),
])
def test_nonexpense_currency_mismatch_and_reverse_time_are_not_removed(parent_fields, refund_fields):
    rows = [row("purchase", **parent_fields), row("refund", "refund", related_id="purchase", **refund_fields)]
    before = deepcopy(rows)
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0
    assert rows == before


@pytest.mark.parametrize("fields", [{"merchant_id": "merchant-order-12345"},
                                    {"description": "可明确识别的商品型号和规格"}])
def test_unique_same_source_full_order_or_meaningful_description_can_link(fields):
    rows = [row("purchase", **fields), row("refund", "refund", **fields)]
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 1


def test_ambiguous_identical_items_do_not_get_greedily_assigned():
    rows = [row("purchase-a", description="相同商品型号及规格"), row("purchase-b", description="相同商品型号及规格"),
            row("refund", "refund", description="相同商品型号及规格")]
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0


def test_wechat_full_refund_status_can_close_complete_two_by_two_group():
    fields = {"source": "wechat", "payment_method": "示例银行(1234)", "status": "已全额退款"}
    rows = [row("purchase-a", direction="out", **fields),
            row("purchase-b", direction="out", **fields),
            row("refund-a", "refund", direction="in", time="2026-03-01 00:00:00", **fields),
            row("refund-b", "refund", direction="in", time="2026-03-02 00:00:00", **fields)]
    audit = remove_fully_refunded(rows)
    assert audit["summary"]["group_count"] == 1
    assert audit["summary"]["record_count"] == 4
    assert audit["summary"]["expense_removed"] == 20000
    assert "组级关联" in audit["groups"][0]["reason"]
    assert remove_fully_refunded(rows) == audit


@pytest.mark.parametrize("change", ["status", "method", "count", "reverse"])
def test_wechat_status_groups_require_complete_correlated_evidence(change):
    fields = {"source": "wechat", "payment_method": "示例银行(1234)", "status": "已全额退款"}
    rows = [row("purchase", direction="out", **fields), row("refund", "refund", direction="in", **fields)]
    if change == "status":
        rows[0]["status"] = "支付成功"
    elif change == "method":
        rows[1]["payment_method"] = "零钱"
    elif change == "count":
        rows.append(row("other-purchase", direction="out", **fields))
    else:
        rows[1]["time"] = "2025-12-31 00:00:00"
    before = deepcopy(rows)
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0
    assert rows == before


def douyin_fixture():
    fields = {"source": "douyin", "payment_method": "抖音月付", "transaction_id": "origin5678",
              "merchant_id": "merchant-original", "direction": "out"}
    confirmation = row("confirmation", **{**fields, "merchant_id": "ONP-confirm"})
    parent = row("purchase", evidence=[deepcopy(confirmation)], **fields)
    refund = row("refund", "refund", **{**fields, "merchant_id": "INP-refund", "direction": "in"})
    chain = {"original_id": "purchase", "original_amount": 10000, "balanced": True,
             "onp_outgoing_ids": ["confirmation"], "onp_refund_ids": [], "inp_refund_ids": ["refund"]}
    return [parent, refund], {"chains": [chain], "ambiguous_events": []}


def test_douyin_full_chain_removes_canonical_purchase_and_refund_once_only():
    rows, chains = douyin_fixture()
    evidence_before = deepcopy(rows[0]["evidence"])
    audit = remove_fully_refunded(rows, douyin_chains=chains)
    assert audit["summary"]["expense_removed"] == audit["summary"]["refund_removed"] == 10000
    assert audit["summary"]["record_count"] == 2
    assert rows[0]["evidence"] == evidence_before
    assert "非完整订单号关联" in audit["groups"][0]["reason"]
    assert remove_fully_refunded(rows, douyin_chains=chains) == audit


def test_douyin_multiple_reversals_can_fully_refund_one_original():
    rows, chains = douyin_fixture()
    rows[0]["evidence"] = []
    rows[1].update(amount=4000, merchant_id="ONP-first")
    rows.append({**rows[1], "id": "second", "amount": 6000, "merchant_id": "ONP-second"})
    chains["chains"][0].update(onp_outgoing_ids=[], onp_refund_ids=["refund", "second"], inp_refund_ids=[])
    assert remove_fully_refunded(rows, douyin_chains=chains)["summary"]["record_count"] == 3


@pytest.mark.parametrize("change", ["partial", "live_confirmation", "wrong_suffix", "currency", "reused", "ambiguous"])
def test_douyin_chain_needs_whole_amount_and_unique_current_evidence(change):
    rows, chains = douyin_fixture()
    if change == "partial":
        rows[1]["amount"] = 2000
    elif change == "live_confirmation":
        rows.append(deepcopy(rows[0]["evidence"][0]))
    elif change == "wrong_suffix":
        rows[1]["transaction_id"] = "different1234"
    elif change == "currency":
        rows[1]["currency"] = "USD"
    elif change == "reused":
        chains["chains"].append(deepcopy(chains["chains"][0]))
    else:
        chains["ambiguous_events"] = [{"candidate_original_ids": ["purchase", "another"]}]
    before = deepcopy(rows)
    assert remove_fully_refunded(rows, douyin_chains=chains)["summary"]["group_count"] == 0
    assert rows == before


def bank_pair():
    fields = {"source": "bank_boc", "account_tail": "1234", "counterparty": "网银在线-示例线缆旗舰店",
              "time_precision": "second", "payment_method": "示例银行卡(1234)"}
    return [row("purchase", amount=2207, direction="out", description="网上快捷支付", raw={"summary": "网上快捷支付"}, **fields),
            row("refund", "refund", amount=2207, direction="in", description="网上快捷退款", raw={"summary": "网上快捷退款"},
                time="2026-01-02 10:02:32", **fields)]


def test_unique_same_bank_account_merchant_refund_removes_both_records():
    rows = bank_pair()
    audit = remove_fully_refunded(rows)
    assert audit["summary"]["expense_removed"] == audit["summary"]["refund_removed"] == 2207
    assert audit["summary"]["record_count"] == 2
    assert "同银行账户" in audit["groups"][0]["reason"]
    assert "无完整订单号" in audit["groups"][0]["reason"]
    assert remove_fully_refunded(rows) == audit


@pytest.mark.parametrize("refund_time", ["2026-01-02 10:00:00", "2026-01-03 10:00:00"])
def test_bank_same_second_and_exactly_24_hours_are_supported(refund_time):
    rows = bank_pair()
    rows[1]["time"] = refund_time
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 1


@pytest.mark.parametrize("change", ["account", "missing_account", "source", "currency", "prefix", "reverse",
                                    "late", "partial", "transfer", "loan", "red_packet", "internal", "coarse_time"])
def test_bank_short_refund_requires_complete_account_and_purchase_evidence(change):
    rows = bank_pair()
    if change == "account":
        rows[1]["account_tail"] = "5678"
    elif change == "missing_account":
        rows[0]["account_tail"] = rows[1]["account_tail"] = ""
    elif change == "source":
        rows[1]["source"] = "bank_cib"
    elif change == "currency":
        rows[1]["currency"] = "USD"
    elif change == "prefix":
        rows[1]["counterparty"] = "银联-示例线缆旗舰店"
    elif change == "reverse":
        rows[1]["time"] = "2026-01-02 09:59:59"
    elif change == "late":
        rows[1]["time"] = "2026-01-03 10:00:01"
    elif change == "partial":
        rows[1]["amount"] = 2206
    elif change in {"transfer", "loan", "red_packet"}:
        rows[0]["description"] = {"transfer": "对外转账", "loan": "借款归还", "red_packet": "红包付款"}[change]
    elif change == "internal":
        rows[0]["kind"] = "internal"
    else:
        rows[0]["time_precision"] = "day"
    before = deepcopy(rows)
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0
    assert rows == before


@pytest.mark.parametrize("party", ["", "付费通(拼多多支付)-拼多多平台商户", "美团支付-美团汉海信息技术有限公司",
                                    "京东商城业务", "网银在线（北京）科技有限公司", "财付通支付科技有限公司"])
def test_bank_generic_payment_platform_is_not_a_specific_merchant(party):
    rows = bank_pair()
    for item in rows:
        item["counterparty"] = party
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0


@pytest.mark.parametrize("extra_kinds", [("expense",), ("refund",), ("expense", "refund")])
def test_bank_candidate_graph_must_be_unique_in_both_directions(extra_kinds):
    rows = bank_pair()
    for index, kind in enumerate(extra_kinds):
        item = deepcopy(rows[0 if kind == "expense" else 1])
        item["id"] = "extra-" + str(index)
        rows.append(item)
    before = deepcopy(rows)
    assert remove_fully_refunded(rows)["summary"]["group_count"] == 0
    assert rows == before
