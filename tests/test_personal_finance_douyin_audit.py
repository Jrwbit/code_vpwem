import unittest

from tools.personal_finance.douyin_audit import annotate_douyin_chains


def row(number, suffix, amount, prefix="200", direction="out", method="抖音月付"):
    return {"id": str(number), "row": number, "source": "douyin",
            "transaction_id": f"100000{number:04}{suffix}", "merchant_id": f"{prefix}{number}",
            "direction": direction, "amount": amount, "payment_method": method,
            "counterparty": "示例商户", "time": f"2026-01-{number:02} 12:00:00",
            "transaction_type": "退款" if direction == "in" else "支付"}


class DouyinAuditTests(unittest.TestCase):
    def test_split_confirmation_and_refund_close_original(self):
        records = [row(1, "1234", 10000), row(2, "1234", 3000, "ONP"),
                   row(3, "1234", 3000, "ONP"), row(4, "1234", 4000, "ONP", "in", "退款")]
        result = annotate_douyin_chains(records)
        self.assertTrue(result["summary"]["all_chains_balanced"])
        self.assertEqual(result["onp_expense_duplicate_candidate_ids"], ["2", "3"])
        self.assertEqual(result["summary"]["candidate_net_expense_after_dedup"], 6000)

    def test_refund_after_confirmation_is_kept(self):
        records = [row(1, "1234", 10000), row(2, "1234", 10000, "ONP"),
                   row(3, "1234", 10000, "INP", "in", "退款")]
        result = annotate_douyin_chains(records)
        self.assertTrue(result["summary"]["all_chains_balanced"])
        self.assertEqual(result["summary"]["inp_refund_amount"], 10000)
        self.assertEqual(result["summary"]["candidate_net_expense_after_dedup"], 0)

    def test_four_digit_collision_is_not_silently_joined(self):
        records = [row(1, "1234", 10000), row(2, "1234", 10000), row(3, "1234", 10000, "ONP")]
        result = annotate_douyin_chains(records)
        self.assertEqual(len(result["ambiguous_events"]), 1)
        self.assertFalse(result["summary"]["all_chains_balanced"])
        self.assertEqual(result["onp_expense_duplicate_candidate_ids"], [])


if __name__ == "__main__":
    unittest.main()
