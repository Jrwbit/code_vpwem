import unittest
from copy import deepcopy

from tools.personal_finance.entities import annotate_entities


def transaction(identifier, party, kind="expense", amount=1000, method="", category="餐饮美食", **kwargs):
    return {"id": identifier, "counterparty": party, "kind": kind, "amount": amount,
            "payment_method": method, "category": category, "evidence": [], **kwargs}


class EntityTests(unittest.TestCase):
    def test_real_name_nickname_and_exact_alias_without_description_matching(self):
        rows = [transaction("a", "王甲 (昵称)", category="转账收入"),
                transaction("b", "王甲", category="本人转账"),
                transaction("c", "面馆", description="王甲推荐的餐馆"),
                transaction("d", "旧称", category="转账收入")]
        directory = annotate_entities(rows, {"旧称": "王甲"})
        person = next(item for item in directory if item["type"] == "person")
        self.assertEqual(person["label"], "王甲")
        self.assertIn("王甲 (昵称)", person["aliases"])
        for row in [rows[0], rows[1], rows[3]]:
            self.assertIn(person["id"], row["entity_ids"])
        self.assertNotIn(person["id"], rows[2]["entity_ids"])

    def test_short_merchants_and_processors_are_not_guessed_as_people(self):
        rows = [transaction("a", "李记"), transaction("b", "支付宝", category="转账收入"),
                transaction("c", "财付通", category="本人转账"),
                transaction("d", "平安一生", category="转账收入")]
        directory = annotate_entities(rows)
        self.assertTrue(all(item["type"] == "merchant" for item in directory))

    def test_unconfirmed_family_payer_stays_separate_and_confirmed_alias_merges(self):
        rows = [transaction("family", "商户", method="亲情卡(甲乙(**乙))"),
                transaction("transfer", "王甲乙", category="转账收入")]
        directory = annotate_entities(rows)
        self.assertEqual(len([item for item in directory if item["type"] == "person"]), 2)
        directory = annotate_entities(rows, {"甲乙": "王甲乙"})
        people = [item for item in directory if item["type"] == "person"]
        self.assertEqual(len(people), 1)
        person = people[0]
        self.assertEqual(rows[0]["entity_roles"][person["id"]], ["payer"])
        self.assertEqual(rows[1]["entity_roles"][person["id"]], ["counterparty"])
        family = next(item for item in directory if item["label"] == "亲情卡")
        self.assertIn("亲戚卡", family["aliases"])
        self.assertTrue(family["featured"])

    def test_refund_inherits_missing_method_through_related_evidence_id(self):
        original = transaction("purchase", "超市", method="亲情卡(甲乙(**乙))")
        original["evidence"] = [{"id": "raw-purchase", "counterparty": "超市", "payment_method": "亲情卡(甲乙(**乙))"}]
        refund = transaction("refund", "超市", kind="refund", related_id="raw-purchase", method="退款")
        directory = annotate_entities([original, refund])
        family_id = next(item["id"] for item in directory if item["label"] == "亲情卡")
        self.assertIn(family_id, refund["entity_ids"])
        self.assertEqual(refund["entity_roles"][family_id], ["payment"])

    def test_merged_evidence_does_not_duplicate_ids_or_change_financial_records(self):
        row = transaction("one", "王甲", category="转账收入")
        row["evidence"] = [deepcopy(row), deepcopy(row)]
        before = deepcopy(row)
        directory = annotate_entities([row])
        self.assertEqual(len(row["entity_ids"]), 1)
        self.assertEqual(len(directory), 1)
        self.assertEqual({key: value for key, value in row.items() if key not in {"entity_ids", "entity_roles"}}, before)

    def test_one_person_can_be_both_counterparty_and_payer_once(self):
        row = transaction("one", "王甲乙", category="本人转账", method="亲情卡(甲乙(**乙))")
        directory = annotate_entities([row], {"甲乙": "王甲乙"})
        person = next(item for item in directory if item["type"] == "person")
        self.assertEqual(row["entity_roles"][person["id"]], ["counterparty", "payer"])
        self.assertEqual(row["entity_ids"].count(person["id"]), 1)

    def test_confirmed_alias_target_is_a_person_without_surname_guessing(self):
        rows = [transaction("payer", "商户", method="亲情卡(付款甲(**甲))"),
                transaction("transfer", "测试甲", category="转账收入"),
                transaction("unrelated", "测试甲商城", description="测试甲付款")]
        directory = annotate_entities(rows, {"付款甲": "测试甲"})
        people = [entry for entry in directory if entry["type"] == "person"]
        self.assertEqual(len(people), 1)
        person = people[0]
        self.assertEqual(person["label"], "测试甲")
        self.assertEqual(rows[0]["entity_roles"][person["id"]], ["payer"])
        self.assertEqual(rows[1]["entity_roles"][person["id"]], ["counterparty"])
        self.assertNotIn(person["id"], rows[2]["entity_ids"])


if __name__ == "__main__":
    unittest.main()
