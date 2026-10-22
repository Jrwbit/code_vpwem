import unittest
from copy import deepcopy

from tools.personal_finance.foreign_audit import annotate_foreign_cash


def record(identifier, currency, direction, amount, time, summary, balance=None, account="example"):
    raw = {"account": account, "summary": summary}
    if balance is not None:
        raw["balance_cents"] = balance
    return {"id": identifier, "source": "bank_example", "currency": currency,
            "direction": direction, "amount": amount, "time": time, "raw": raw,
            "description": summary, "kind": "internal", "review": False, "evidence": []}


class ForeignCashAuditTests(unittest.TestCase):
    def records(self):
        cny = record("cny", "CNY", "out", 45678, "2026-08-26 11:22:00", "购汇")
        purchase = record("idr-credit", "IDR", "in", 120000000, "2026-08-26 11:22:18", "购汇", 120000000)
        withdrawal = record("idr-cash", "IDR", "out", 120000000, "2026-08-26 11:27:38", "现金取款", 0)
        return [cny], [purchase, withdrawal]

    def test_marks_only_cny_amount_for_review_and_preserves_original_currency(self):
        transactions, foreign = self.records()
        original_foreign = deepcopy(foreign)
        result = annotate_foreign_cash(transactions, foreign)
        cny = transactions[0]
        self.assertEqual(cny["kind"], "review")
        self.assertEqual(cny["category"], "外币现金待核")
        self.assertTrue(cny["review"])
        self.assertEqual(cny["amount"], 45678)
        self.assertEqual(result["matched_cash_cny_cents"], 45678)
        self.assertEqual(result["foreign_cash_by_currency"], {"IDR": 120000000})
        self.assertEqual([row["currency"] for row in cny["evidence"]], ["IDR", "IDR"])
        self.assertEqual(foreign, original_foreign)
        cny["evidence"][0]["amount"] = 1
        self.assertEqual(foreign, original_foreign)

    def test_repeat_annotation_does_not_duplicate_evidence(self):
        transactions, foreign = self.records()
        annotate_foreign_cash(transactions, foreign)
        result = annotate_foreign_cash(transactions, foreign)
        self.assertEqual(len(transactions[0]["evidence"]), 2)
        self.assertEqual(result["matched_cash_count"], 1)

    def test_foreign_transfer_is_not_cash_withdrawal(self):
        transactions, foreign = self.records()
        foreign[1]["raw"]["summary"] = "转账"
        self.assertEqual(annotate_foreign_cash(transactions, foreign)["matched_cash_count"], 0)
        self.assertEqual(transactions[0]["kind"], "internal")

    def test_ambiguous_purchase_pairing_does_not_reclassify(self):
        transactions, foreign = self.records()
        transactions.append({**deepcopy(transactions[0]), "id": "other-cny"})
        self.assertEqual(annotate_foreign_cash(transactions, foreign)["matched_cash_count"], 0)
        self.assertTrue(all(row["kind"] == "internal" for row in transactions))

    def test_partial_or_late_cash_and_different_accounts_are_not_inferred(self):
        for field, value in (("amount", 60000000), ("time", "2026-08-26 12:27:38"), ("account", "different")):
            with self.subTest(field=field):
                transactions, foreign = self.records()
                if field == "account":
                    foreign[1]["raw"][field] = value
                else:
                    foreign[1][field] = value
                self.assertEqual(annotate_foreign_cash(transactions, foreign)["matched_cash_count"], 0)

    def test_intervening_currency_activity_or_inconsistent_balance_blocks_link(self):
        transactions, foreign = self.records()
        foreign.append(record("other", "IDR", "out", 100, "2026-08-26 11:25:00", "转账"))
        self.assertEqual(annotate_foreign_cash(transactions, foreign)["matched_cash_count"], 0)
        transactions, foreign = self.records()
        foreign[1]["raw"]["balance_cents"] = 1
        self.assertEqual(annotate_foreign_cash(transactions, foreign)["matched_cash_count"], 0)

    def test_unrelated_foreign_purchase_is_not_reclassified(self):
        transactions, foreign = self.records()
        foreign[0]["time"] = "2026-08-26 11:23:00"
        self.assertEqual(annotate_foreign_cash(transactions, foreign)["matched_cash_count"], 0)


if __name__ == "__main__":
    unittest.main()
