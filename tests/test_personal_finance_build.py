import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.personal_finance.build import apply_decisions, build_data, render_dashboard, write_outputs


def bank_record(identifier, currency="CNY"):
    return {"id": identifier, "source": "bank_boc", "source_label": "中国银行", "row": 1,
            "time": "2026-01-02 03:04:05", "amount": 1234, "currency": currency,
            "direction": "out", "counterparty": "示例商户", "description": "商品",
            "payment_method": "中国银行", "status": "已入账", "transaction_id": identifier,
            "merchant_id": "", "raw": {}}


class BuildTests(unittest.TestCase):
    @patch("tools.personal_finance.build.parse_alipay")
    def test_categories_and_person_aliases_preserve_reconciled_amounts(self, parser):
        def row(identifier, order, when, amount, direction, party, description, method):
            return {**bank_record(identifier), "source": "alipay", "source_label": "支付宝",
                    "transaction_id": order, "time": when, "amount": amount, "direction": direction,
                    "counterparty": party, "description": description, "payment_method": method}
        purchase = row("purchase", "order1", "2026-01-02 12:00:00", 10000, "out",
                       "示例酒店", "酒店住宿房费", "亲情卡(付款甲(**甲))")
        refund = row("refund", "order1_refund1", "2026-02-03 12:00:00", 2000, "neutral",
                     "示例酒店", "退款", "退款")
        refund["status"] = "退款成功"
        transfer = row("transfer", "order2", "2026-03-04 12:00:00", 30000, "in",
                       "测试甲", "转账收入", "余额")
        transfer["transaction_type"] = "转账"
        parser.return_value = ([purchase, refund, transfer], {"source": "alipay"})
        with tempfile.TemporaryDirectory() as directory:
            original, decrypted = Path(directory) / "original", Path(directory) / "decrypted"
            original.mkdir(); decrypted.mkdir()
            (original / "支付宝.csv").write_text("synthetic input")
            data = build_data(original, decrypted, "2026-10-22", person_aliases={"付款甲": "测试甲"})
        self.assertEqual(sum(month["income"] for month in data["monthly"].values()), 30000)
        self.assertEqual(sum(month["net_expense"] for month in data["monthly"].values()), 8000)
        transactions = {row["transaction_id"]: row for row in data["transactions"]}
        self.assertEqual(transactions["order1"]["spending_category"], "住宿")
        self.assertEqual(transactions["order1_refund1"]["spending_category"], "住宿")
        person = next(entity for entity in data["entities"] if entity["label"] == "测试甲")
        self.assertTrue(all(person["id"] in row["entity_ids"] for row in data["transactions"]))
        self.assertEqual(len(data["transactions"]), 3)

    def test_user_decision_can_uniquely_match_merged_evidence(self):
        evidence = bank_record("bank-evidence")
        transaction = {**bank_record("canonical"), "source": "wechat", "kind": "expense",
                       "category": "转账", "review": True, "evidence": [evidence]}
        decisions = [{"match": {"source": "bank_boc", "date": "2026-01-02", "amount": 1234,
                                  "direction": "out", "counterparty": "示例商户"},
                      "kind": "internal", "category": "借款本金", "reason": "已确认是归还本金"}]
        applied = apply_decisions([transaction], decisions)
        self.assertEqual(transaction["kind"], "internal")
        self.assertEqual(transaction["category"], "借款本金")
        self.assertFalse(transaction["review"])
        self.assertEqual(transaction["review_reason"], "")
        self.assertEqual(transaction["reason"], "用户确认：已确认是归还本金")
        self.assertEqual(applied[0]["record_id"], "canonical")
        self.assertEqual(applied[0]["selector"], decisions[0]["match"])

    def test_user_decisions_require_unique_matches_and_apply_atomically(self):
        first = {**bank_record("first"), "kind": "expense", "review": True}
        second = {**bank_record("second"), "kind": "expense", "review": True}
        selector = {"source": "bank_boc", "date": "2026-01-02", "amount": 1234, "direction": "out"}
        decision = {"match": selector, "kind": "internal", "reason": "归还本金"}
        with self.assertRaisesRegex(ValueError, "matched 2"):
            apply_decisions([first, second], [decision])
        with self.assertRaisesRegex(ValueError, "matched 0"):
            apply_decisions([], [decision])
        with self.assertRaisesRegex(ValueError, "conflicting kinds"):
            apply_decisions([first], [decision, {**decision, "kind": "income"}])
        self.assertEqual(first["kind"], "expense")
        self.assertTrue(first["review"])

    def test_script_payload_cannot_end_script_element(self):
        payload = {"name": '</script><script>alert("x")</script>', "note": "&\u2028\u2029"}
        html = render_dashboard(payload, '<script>window.FINANCE_DATA=/*__FINANCE_DATA__*/ null;</script>')
        self.assertEqual(html.count("</script>"), 1)
        encoded = html.split("window.FINANCE_DATA=", 1)[1].removesuffix(";</script>")
        self.assertEqual(json.loads(encoded), payload)

    @patch("tools.personal_finance.build.parse_bank_pdf")
    def test_foreign_rows_are_separate_and_only_decrypted_pdfs_are_read(self, parser):
        parser.return_value = ([bank_record("cny"), bank_record("usd", "USD")],
                               {"source": "bank_boc", "period_start": "2026-01-01", "period_end": "2026-10-22"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, decrypted, output = (root / name for name in ("original", "decrypted", "output"))
            original.mkdir(); decrypted.mkdir()
            (original / "encrypted.pdf").write_bytes(b"encrypted input is ignored")
            source = decrypted / "bank.pdf"
            source.write_bytes(b"synthetic bank PDF")
            data = build_data(original, decrypted, "2026-10-22")
            parser.assert_called_once_with(source)
            self.assertEqual(len(data["transactions"]), 1)
            self.assertEqual(len(data["foreign_transactions"]), 1)
            self.assertEqual(data["monthly"]["2026-01"]["expense"], 1234)
            self.assertTrue(data["meta"]["incomplete"])
            self.assertEqual(len(data["meta"]["missing_sources"]), 5)
            write_outputs(data, output)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            for name in ("finance-data.json", "index.html", "monthly.csv", "audit.json"):
                self.assertEqual(stat.S_IMODE((output / name).stat().st_mode), 0o600)
            self.assertEqual(json.loads((output / "finance-data.json").read_text())["audit"]["foreign_count"], 1)

    @patch("tools.personal_finance.build.parse_alipay", side_effect=ValueError("invalid synthetic header"))
    def test_failed_parser_is_reported_and_source_remains_missing(self, parser):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, decrypted = root / "original", root / "decrypted"
            original.mkdir(); decrypted.mkdir()
            (original / "支付宝.csv").write_text("synthetic invalid file")
            data = build_data(original, decrypted, "2026-10-22")
        self.assertEqual(len(data["errors"]), 1)
        self.assertIn("支付宝", data["meta"]["missing_sources"])
        self.assertEqual(data["transactions"], [])
        self.assertEqual(data["errors"][0]["message"], "invalid synthetic header")


if __name__ == "__main__":
    unittest.main()
