import unittest
from copy import deepcopy

from tools.personal_finance.self_transfer_audit import audit_self_transfers


def leg(identifier, source, account, counterparty, direction, time, amount=100000, target_field="counterparty_account"):
    return {"id": identifier, "source": source, "source_label": source,
            "currency": "CNY", "direction": direction, "time": time,
            "amount": amount, "row": 7, "kind": "internal", "category": "本人转账",
            "raw": {"account": account, target_field: counterparty}, "evidence": [{"id": identifier + "-original"}]}


class SelfTransferAuditTests(unittest.TestCase):
    def pair(self):
        return [
            leg("out", "bank_a", "11111111111", "22222222222收款银行", "out", "2026-01-01 22:00:00", target_field="counterparty_account_bank"),
            leg("in", "bank_b", "22222222222", "测试人 11111111111", "in", "2026-01-02 00:00:00", target_field="counterparty_info"),
        ]

    def test_pair_is_counted_once_and_inputs_remain_unchanged(self):
        rows = self.pair()
        before = deepcopy(rows)
        audit = audit_self_transfers(rows)
        self.assertEqual(audit["summary"]["pair_count"], 1)
        self.assertEqual(audit["summary"]["amount_cents"], 100000)
        self.assertEqual(audit["summary"]["matched_leg_count"], 2)
        self.assertEqual(audit["summary"]["matched_leg_amount_cents"], 200000)
        self.assertEqual(audit["single_self_legs"], [])
        self.assertEqual(rows, before)
        match = audit["matches"][0]
        self.assertEqual(match["type"], "internal_transfer")
        self.assertEqual(match["record_ids"], ["out", "in"])
        self.assertIn("2026-01-01 第7行", match["reason"])
        self.assertNotIn("11111111111", match["reason"])
        self.assertNotIn("22222222222", match["reason"])

    def test_one_way_account_evidence_is_insufficient(self):
        rows = self.pair()
        rows[1]["raw"]["counterparty_info"] = "未提供账号"
        audit = audit_self_transfers(rows)
        self.assertEqual(audit["summary"]["pair_count"], 0)
        self.assertEqual(audit["summary"]["single_self_leg_count"], 2)
        self.assertTrue(all(row["kind"] == "internal" for row in rows))

    def test_multiple_equal_candidates_are_not_greedily_paired(self):
        rows = self.pair()
        rows.append({**deepcopy(rows[1]), "id": "another-in"})
        audit = audit_self_transfers(rows)
        self.assertEqual(audit["matches"], [])
        self.assertEqual(audit["summary"]["single_self_leg_count"], 3)
        rows = self.pair()
        rows.append({**deepcopy(rows[0]), "id": "another-out"})
        self.assertEqual(audit_self_transfers(rows)["matches"], [])

    def test_different_amount_direction_or_late_posting_is_unmatched(self):
        for field, value in (("amount", 100001), ("direction", "out"), ("time", "2026-01-03 00:00:00")):
            with self.subTest(field=field):
                rows = self.pair()
                rows[1][field] = value
                self.assertEqual(audit_self_transfers(rows)["matches"], [])

    def test_only_explicit_internal_self_bank_cny_rows_qualify(self):
        for field, value in (("source", "alipay"), ("kind", "income"), ("category", "投资流转"), ("currency", "USD")):
            with self.subTest(field=field):
                rows = self.pair()
                rows[1][field] = value
                audit = audit_self_transfers(rows)
                self.assertEqual(audit["matches"], [])
                self.assertEqual(audit["summary"]["self_leg_count"], 1)

    def test_account_substring_does_not_establish_ownership(self):
        rows = self.pair()
        rows[0]["raw"]["counterparty_account_bank"] = "9222222222220"
        self.assertEqual(audit_self_transfers(rows)["matches"], [])


if __name__ == "__main__":
    unittest.main()
