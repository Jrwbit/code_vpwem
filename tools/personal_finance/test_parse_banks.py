"""Synthetic fixtures only: no personal transaction data is stored here."""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from parse_banks import BankStatementError, _audit, _parse_boc, _parse_cib, _parse_cmb, _validate_balances, money_cents


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float


def line(text, x, y, direction=(1, 0)):
    return {"dir": direction, "bbox": (x, y, x + 1, y + 1), "spans": [{"text": text}]}


class FakePage:
    """Model PDF lines and the first-cell box plus vertical column edges."""

    def __init__(self, rows, totals=None):
        self.lines = []
        self.drawings = []
        if totals:
            self.lines += [line(totals, 50, 10), line("2026-01-01-2026-10-22", 50, 20), line("账户账号：00001234", 50, 30)]
        boundaries = [50, 100.86, 151.71, 202.57, 246.64, 297.5, 348.36, 399.21, 450.07, 545]
        for n, row in enumerate(rows):
            y = 301 + n * 40
            self.drawings.append({"rect": Box(50, y, 100.86, y + 40)})
            for right in boundaries[2:]:
                self.drawings.append({"rect": Box(right, y, right, y + 40)})
            self.lines += [line(row[0][:10], 55, y + 10), line(row[0][10:], 55, y + 20)]
            for i, cell in enumerate(row[1:], 1):
                if cell:
                    self.lines.append(line(cell, boundaries[i] + 2, y + 15))
            # A watermark overlapping the amount column must never enter a cell.
            self.lines.append(line("2026 银行 9999", 250, y + 14, direction=(0.7, -0.7)))

    def get_text(self, mode):
        return {"blocks": [{"lines": self.lines}]}

    def get_drawings(self):
        return self.drawings


class BankParserPrimitivesTests(unittest.TestCase):
    def test_currency_conversion_is_exact(self):
        self.assertEqual(money_cents("1,234.56"), 123456)
        self.assertEqual(money_cents("-0.01"), -1)
        self.assertEqual(money_cents("(￥4.20)"), -420)

    def test_invalid_precision_is_rejected(self):
        for value in ("0.001", "NaN", "Infinity", "not money"):
            with self.subTest(value=value), self.assertRaises(BankStatementError):
                money_cents(value)

    def test_audit_accounts_for_each_page(self):
        records = [
            {"raw": {"page": 1}, "direction": "in", "amount": 101, "time": "2026-01-01 00:00:00"},
            {"raw": {"page": 2}, "direction": "out", "amount": 99, "time": "2026-01-02 00:00:00"},
            {"raw": {"page": 2}, "direction": "out", "amount": 1, "time": "2026-01-02 01:00:00"},
        ]
        audit = _audit(records, 3)
        self.assertEqual(audit["income_cents"], 101)
        self.assertEqual(audit["expense_cents"], 100)
        self.assertEqual([p["rows"] for p in audit["page_totals"]], [1, 2, 0])

    def test_currencies_are_never_summed_as_renminbi(self):
        records = [
            {"raw": {"page": 1}, "direction": "out", "amount": 1000, "currency": "CNY", "time": "2026-01-01 00:00:00"},
            {"raw": {"page": 1}, "direction": "out", "amount": 99999900, "currency": "IDR", "time": "2026-01-01 00:00:00"},
        ]
        audit = _audit(records, 1)
        self.assertEqual(audit["expense_cents"], 1000)
        self.assertEqual(audit["totals_by_currency"]["IDR"]["expense_cents"], 99999900)

    def test_balance_check_separates_currency_accounts(self):
        records = [
            {"id": "c1", "currency": "CNY", "direction": "in", "amount": 1000, "raw": {"balance_cents": 1000}},
            {"id": "u1", "currency": "USD", "direction": "in", "amount": 200, "raw": {"balance_cents": 200}},
            {"id": "c2", "currency": "CNY", "direction": "out", "amount": 100, "raw": {"balance_cents": 900}},
        ]
        self.assertEqual(_validate_balances(records), 1)


class IndustrialBankTests(unittest.TestCase):
    def statement(self, totals="支出 ¥2.00(1)笔     收入 ¥3.00(1)笔", balance="11.00"):
        return [
            FakePage([["2026-01-0112:00:00", "20260101", "快捷支付", "支", "-2.00", "8.00", "订单123,商户", "商户", "支付机构"]], totals),
            FakePage([["2026-01-0213:00:00", "20260102", "退款", "收", "3.00", balance, "退款", "商户", "支付机构"]]),
        ]

    def test_multiple_pages_totals_balances_and_watermarks(self):
        records, audit = _parse_cib(self.statement())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["amount"], 200)
        self.assertEqual(records[0]["merchant_id"], "123")
        self.assertEqual(records[1]["raw"]["page"], 2)
        self.assertEqual(records[1]["account_tail"], "1234")
        self.assertTrue(audit["printed_totals_match"])
        self.assertEqual(audit["balance_checks"], 1)

    def test_printed_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(BankStatementError, "printed totals mismatch"):
            _parse_cib(self.statement(totals="支出 ¥2.00(2)笔     收入 ¥3.00(1)笔"))

    def test_balance_discontinuity_is_rejected(self):
        with self.assertRaisesRegex(BankStatementError, "balance discontinuities"):
            _parse_cib(self.statement(balance="11.01"))

    def test_unparsed_date_anchor_is_rejected(self):
        statement = self.statement()
        statement[1].lines.append(line("2026-01-03", 55, 400))
        with self.assertRaisesRegex(BankStatementError, "parsed 1/2 date rows"):
            _parse_cib(statement)


class ChinaMerchantsBankTests(unittest.TestCase):
    def statement(self):
        page = FakePage([])
        page.lines = [line("账号：00009999", 261, 100), line("2026-01-01 -- 2026-10-20", 238, 90)]
        for y, cells in [(300, ["2026-01-01", "CNY", "10.00", "10.00", "汇入汇款", "测试学校123456789"]), (327, ["2026-01-02", "CNY", "-2.00", "8.00", "快捷支付", "测试商户123456789"])]:
            for x, text in zip([36, 98.75, 156.28, 234.73, 307.89, 417.65], cells):
                page.lines.append(line(text, x, y))
        return [page]

    def test_date_only_counterparty_and_balances(self):
        records, audit = _parse_cmb(self.statement())
        self.assertEqual(records[0]["counterparty"], "测试学校")
        self.assertEqual(records[0]["time_precision"], "day")
        self.assertEqual(records[0]["time"], "2026-01-01 00:00:00")
        self.assertEqual(audit["balance_checks"], 1)
        self.assertEqual(audit["income_cents"], 1000)


class FakeBocPage(FakePage):
    def __init__(self, rows, totals, row_count=None):
        self.lines = [line("借记卡号：00008888", 30, 50), line("交易区间：2026-01-01 至2026-10-22", 30, 60),
                      line(f"借方发生数：{totals[0]}", 200, 94), line(f"贷方发生数：{totals[1]}", 365, 94),
                      line(f"行数:{len(rows) if row_count is None else row_count}", 573, 94)]
        xs = [30, 98, 155, 210, 273, 338, 393, 445, 517, 592, 665, 737, 820]
        ys = [129.7, 145.7] + [163.7 + i * 18 for i in range(len(rows))]
        self.drawings = []
        for x in xs:
            self.drawings.append({"items": [("l", SimpleNamespace(x=x, y=129), SimpleNamespace(x=x, y=146))]})
        for y in ys:
            self.drawings.append({"items": [("l", SimpleNamespace(x=30, y=y), SimpleNamespace(x=98, y=y))]})
        self.raw_lines = []
        for row_number, row in enumerate(rows):
            # All columns intentionally share one PDF line, like BOC's merged
            # remark/counterparty spans. Only character geometry separates them.
            chars = []
            y = 148 + row_number * 18
            for x, cell in zip(xs, row):
                for character_number, character in enumerate(cell):
                    xx = x + 2 + character_number
                    chars.append({"c": character, "bbox": (xx, y, xx + 1, y + 1)})
            self.raw_lines.append({"dir": (1, 0), "spans": [{"chars": chars}]})

    def get_text(self, mode):
        return {"blocks": [{"lines": self.raw_lines if mode == "rawdict" else self.lines}]}


class BankOfChinaTests(unittest.TestCase):
    def statement(self, row_count=None):
        # The input statement runs newest-first and interleaves currencies.
        rows = [
            ["2026-01-03", "12:00:00", "人民币", "-2.00", "8.00", "网上快捷支付", "银企对接", "---", "附言甲", "商户乙", "123", "银行"],
            ["2026-01-02", "12:00:00", "美元", "5.00", "5.00", "购汇", "手机银行", "---", "", "本人", "123", "银行"],
            ["2026-01-01", "12:00:00", "人民币", "10.00", "10.00", "转账", "手机银行", "---", "", "学校", "123", "银行"],
        ]
        return [FakeBocPage(rows, ("2.00", "15.00"), row_count=row_count)]

    def test_character_cells_page_totals_and_currency(self):
        records, audit = _parse_boc(self.statement())
        self.assertEqual(records[0]["counterparty"], "商户乙")
        self.assertEqual(records[0]["raw"]["remark"], "附言甲")
        self.assertEqual(records[0]["raw"]["branch"], "")
        self.assertEqual(audit["income_cents"], 1000)
        self.assertEqual(audit["totals_by_currency"]["USD"]["income_cents"], 500)
        self.assertEqual(audit["balance_checks"], 1)
        self.assertTrue(audit["printed_totals_match"])

    def test_printed_page_row_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(BankStatementError, "table grid does not match"):
            _parse_boc(self.statement(row_count=4))


if __name__ == "__main__":
    unittest.main()
