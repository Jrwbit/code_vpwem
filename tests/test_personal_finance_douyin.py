"""Synthetic PDF-coordinate fixtures for wrapped Douyin statement cells."""

import unittest
from xml.sax.saxutils import escape

from tools.personal_finance.parse_douyin import parse_douyin_bbox


class DouyinParserTests(unittest.TestCase):
    def test_wrapped_cells_keep_full_identifiers(self):
        words = [
            (78, 30, "交易时间"), (156, 30, "收/支/其他"), (223, 30, "金额（元）"),
            (309, 30, "收/付款方式"), (448, 30, "交易单号"), (599, 30, "交易对方"), (737, 30, "商家单号"),
            (55, 50, "2026-01-02"), (105, 50, "03:04:05"), (156, 50, "支出"), (223, 50, "12.34"),
            (309, 50, "中国银行借记卡（1"), (309, 60, "234）"), (448, 50, "12345678901234567890"),
            (599, 50, "测试商户"), (737, 50, "987654321098765432"), (737, 60, "1234"),
            (55, 90, "2026-01-03"), (105, 90, "03:04:05"), (156, 90, "收入"), (223, 90, "2.34"),
            (309, 90, "退款"), (448, 90, "12345678901234567891"), (599, 90, "测试商户"),
            (737, 90, "INP1234567890"), (737, 100, "9999"),
            (418, 559, "1/"), (431, 559, "20"),
        ]
        xml = '<html xmlns="http://www.w3.org/1999/xhtml"><body><doc><page width="842" height="595">'
        xml += ''.join(f'<word xMin="{x-5}" xMax="{x+5}" yMin="{y}" yMax="{y+10}">{escape(text)}</word>' for x, y, text in words)
        xml += '</page></doc></body></html>'
        records, metadata = parse_douyin_bbox(xml)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["payment_method"], "中国银行借记卡（1234）")
        self.assertEqual(records[0]["merchant_id"], "9876543210987654321234")
        self.assertEqual(records[1]["merchant_id"], "INP12345678909999")
        self.assertEqual(records[1]["transaction_id"], "12345678901234567891")
        self.assertEqual(records[1]["transaction_type"], "退款")
        self.assertEqual(records[1]["amount"], 234)
        self.assertEqual(metadata["observed_totals"]["out"]["amount"], 1234)
        self.assertIsNone(metadata["totals_match"])


if __name__ == "__main__":
    unittest.main()
