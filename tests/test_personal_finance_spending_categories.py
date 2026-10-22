from copy import deepcopy

from tools.personal_finance.spending_categories import annotate_spending_categories, summarize_spending_categories


def record(identifier, description="", counterparty="平台商户", **fields):
    row = {"id": identifier, "kind": "expense", "amount": 1000, "category": "其他",
           "description": description, "counterparty": counterparty, "evidence": []}
    row.update(fields)
    return row


def test_order_purpose_overrides_ambiguous_platform_and_source_category():
    rows = [record("hotel", "示例酒店订单", transaction_type="酒店旅游"),
            record("meal", "酒店双人自助餐", transaction_type="酒店旅游"),
            record("tour", "海岛包车一日游", transaction_type="酒店旅游"),
            record("park", "【直播间彩妆专场】主题乐园双人套票", transaction_type="酒店旅游"),
            record("card", "支付宝充值校园卡", counterparty="示例大学", transaction_type="教育培训"),
            record("medical", "医保收费支付", counterparty="示例大学", transaction_type="教育培训"),
            record("platform", "订单号12345", counterparty="美团"),
            record("unknown-holiday", "旅行订单号12345", counterparty="旅游平台", transaction_type="酒店旅游")]
    annotate_spending_categories(rows)
    assert [row["spending_category"] for row in rows] == ["住宿", "餐饮", "休闲旅游", "休闲旅游", "生活缴费", "医疗健康", "其他待分", "其他待分"]
    assert rows[4]["spending_category_review"] is True


def test_refund_inherits_original_even_when_link_uses_merged_evidence_id():
    rows = [record("purchase", "标准洗", evidence=[{"id": "bank-original", "description": "快捷支付"}]),
            record("refund", "平台退款", kind="refund", related_id="bank-original"),
            record("unlinked", "平台退款", kind="refund", counterparty="美团"),
            record("specific", "退款-地铁票", kind="refund"),
            record("aa", "聚餐", kind="reimbursement")]
    annotate_spending_categories(rows)
    assert [row["spending_category"] for row in rows] == ["生活缴费", "生活缴费", "其他待分", "交通", "其他待分"]
    assert "继承" in rows[1]["spending_category_reason"]


def test_annotation_changes_no_accounting_evidence_or_original_categories():
    rows = [record("expense", "火车票", amount=5000), record("refund", "退款-火车票", kind="refund", amount=1000),
            record("income", "工资", kind="income", amount=20000), record("aa", kind="reimbursement", amount=500)]
    original = deepcopy(rows)
    result = annotate_spending_categories(rows)
    assert len(rows) == len(original)
    for before, after in zip(original, rows):
        assert {key: value for key, value in after.items() if not key.startswith("spending_category")} == before
    assert result["summary"]["net_expense"] == 3500
    assert sum(category["expense"] for category in result["categories"]) == 5000
    assert sum(category["refund"] for category in result["categories"]) == 1000


def test_clear_merged_evidence_can_resolve_generic_order():
    rows = [record("order", "订单号123", counterparty="美团", evidence=[{"id": "detail", "description": "药房购药", "counterparty": "药房"}]),
            record("cash", "购汇", category="境外现金消费", user_confirmed=True),
            record("transfer", "转账", category="对外转账消费", user_confirmed=True)]
    annotate_spending_categories(rows)
    assert [row["spending_category"] for row in rows] == ["医疗健康", "境外现金", "转账赠与"]
    assert rows[2]["spending_category_review"] is True


def test_address_and_product_mentions_do_not_override_actual_purchase():
    rows = [record("bus", "公交站附近美容门口站台往市中心"),
            record("furniture", "家用酒店折叠移动屏风隔断墙", transaction_type="日用百货"),
            record("package", "会员套餐订单"), record("data", "通信流量套餐"),
            record("shipping", counterparty="顺丰速运")]
    annotate_spending_categories(rows)
    assert [row["spending_category"] for row in rows] == ["交通", "日用购物", "其他待分", "生活缴费", "生活缴费"]


def test_rerun_recomputes_refund_after_original_purpose_changes():
    rows = [record("purchase", "火车票"), record("refund", kind="refund", related_id="purchase")]
    annotate_spending_categories(rows)
    assert rows[1]["spending_category"] == "交通"
    rows[0]["description"] = "酒店订单"
    annotate_spending_categories(rows)
    assert rows[1]["spending_category"] == "住宿"


def test_refund_does_not_inherit_nonexpense_parent_and_detergent_is_goods():
    rows = [record("internal", "药房转账", kind="internal"),
            record("refund", "平台退款", kind="refund", related_id="internal"),
            record("detergent", "洗衣液补充装")]
    annotate_spending_categories(rows)
    assert rows[1]["spending_category"] == "其他待分"
    assert "并非支出原单" in rows[1]["spending_category_reason"]
    assert rows[1]["spending_category_review"] is True
    assert rows[2]["spending_category"] == "日用购物"


def test_specific_retailers_are_recognized_but_university_or_platform_is_not_guessed():
    rows = [record("electronics", counterparty="Apple 示例商场"),
            record("camera", counterparty="大疆旗舰店"),
            record("clothes", counterparty="UNIQLO"),
            record("university", "一网通支付", counterparty="示例大学结算部"),
            record("platform", "订单123", counterparty="抖音生活服务")]
    annotate_spending_categories(rows)
    assert [row["spending_category"] for row in rows] == ["数码家电", "数码家电", "服饰美容", "其他待分", "其他待分"]


def test_summary_after_full_refund_exclusion_preserves_existing_labels():
    rows = [record("purchase", "火车票", amount=5000),
            record("refund", kind="refund", amount=5000, related_id="purchase"),
            record("retained", "酒店订单", amount=8000)]
    before = annotate_spending_categories(rows)
    labels = [{key: value for key, value in item.items() if key.startswith("spending_category")} for item in rows]
    rows[0]["kind"] = rows[1]["kind"] = "excluded"
    after = summarize_spending_categories(rows)
    assert before["summary"]["net_expense"] == after["summary"]["net_expense"] == 8000
    assert after["summary"]["spending_count"] == 1
    assert sum(item["expense"] for item in after["categories"]) == 8000
    assert sum(item["refund"] for item in after["categories"]) == 0
    assert [{key: value for key, value in item.items() if key.startswith("spending_category")} for item in rows] == labels
