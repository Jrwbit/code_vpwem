"""Synthetic fixtures only: real statement contents never enter the repository."""

import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

from tools.personal_finance.parse_apps import cents, find_refund_links, parse_alipay, parse_wechat


class AppParserTests(unittest.TestCase):
    def test_cents_are_exact(self):
        self.assertEqual(cents("￥1,234.56"), 123456)
        self.assertEqual(cents("0.01"), 1)
        with self.assertRaises(ValueError):
            cents("0.001")

    def test_refund_links_normalize_both_order_suffix_separators(self):
        def record(identifier, order, status, row):
            return {"id": identifier, "source": "alipay", "transaction_id": order,
                    "status": status, "row": row, "amount": 100}
        records = [record("payment", "12345_payment", "支付成功", 20),
                   record("refund_star", "12345*refund1", "退款成功", 21),
                   record("refund_underscore", "12345_refund2", "退款成功", 22),
                   record("outside", "67890_refund3", "退款成功", 23)]
        links = find_refund_links(records)
        self.assertEqual([link["original_id"] for link in links], ["payment", "payment"])
        self.assertEqual([link["refund_id"] for link in links], ["refund_star", "refund_underscore"])

    def test_alipay_net_header_matches_closed_refunded_payment(self):
        content = "\n".join([
            "共3笔记录", "支出：2笔 15.00元", "不计收支：1笔 10.00元",
            "交易时间,交易分类,交易对方,对方账号,商品说明,收/支,金额,收/付款方式,交易状态,交易订单号,商家订单号,备注,",
            "2026-01-02 03:04:05,日用百货,商家,/,已退订单,支出,10.00,银行卡(1234),交易关闭,ORDER1,M1,,",
            "2026-02-03 03:04:05,退款,商家,/,商品退款,不计收支,10.00,银行卡(1234),退款成功,ORDER1*refund,M1,,",
            "2026-02-04 03:04:05,日用百货,商家,/,商品,支出,15.00,银行卡(1234),交易成功,ORDER2,M2,,",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statement.csv"
            path.write_text(content, encoding="utf-8-sig")
            records, metadata = parse_alipay(path)
        self.assertFalse(metadata["raw_totals_match"])
        self.assertTrue(metadata["totals_match"])
        self.assertEqual(metadata["reconciled_totals"]["out"]["amount"], 1500)
        self.assertEqual(records[0]["amount"], 1000)
        self.assertEqual(metadata["linked_refund_count"], 1)

    def test_alipay_preserves_neutral_refund_and_closed_status(self):
        content = "\n".join([
            "起始时间：[2026-01-01 00:00:00]    终止时间：[2026-10-22 23:59:59]",
            "共3笔记录", "支出：1笔 12.34元", "不计收支：2笔 22.34元",
            "交易时间,交易分类,交易对方,对方账号,商品说明,收/支,金额,收/付款方式,交易状态,交易订单号,商家订单号,备注,",
            '2026-01-02 03:04:05,日用百货,商家,/,"商品,含逗号",支出,12.34,银行卡(1234),交易成功,12345678901234567890\t,M1\t,,',
            "2026-01-03 03:04:05,退款,商家,/,商品退款,不计收支,12.34,银行卡(1234),退款成功,R1\t,M1\t,,",
            "2026-01-04 03:04:05,投资理财,余额宝,/,银行卡定时转入,不计收支,10.00,,交易关闭,C1\t,M2\t,,",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statement.csv"
            path.write_bytes(content.encode("gb18030"))
            records, metadata = parse_alipay(path)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["transaction_id"], "12345678901234567890")
        self.assertEqual(records[0]["description"], "商品,含逗号")
        self.assertEqual(records[1]["direction"], "neutral")
        self.assertEqual(records[1]["status"], "退款成功")
        self.assertEqual(records[2]["status"], "交易关闭")
        self.assertTrue(metadata["totals_match"])
        self.assertEqual(metadata["period"]["end"], "2026-10-22 23:59:59")

    def test_wechat_excel_time_shared_strings_and_sparse_cells(self):
        header = ["交易时间", "交易类型", "交易对方", "商品", "收/支", "金额(元)",
                  "支付方式", "当前状态", "交易单号", "商户单号", "备注"]
        strings = ["共1笔记录", "支出：1笔 0.30元"] + header + ["商户消费", "商家", "支出", "支付成功", "12345678901234567890"]
        def shared(reference, value):
            return f'<c r="{reference}" t="s"><v>{strings.index(value)}</v></c>'
        rows = [f'<row r="1">{shared("A1", strings[0])}</row>',
                f'<row r="3">{shared("A3", strings[1])}</row>',
                '<row r="5">' + ''.join(shared(f'{chr(65+i)}5', value) for i, value in enumerate(header)) + '</row>',
                '<row r="6"><c r="A6"><v>46317.58708333333</v></c>' + shared("B6", "商户消费") + shared("C6", "商家") + shared("E6", "支出") + '<c r="F6"><v>0.3</v></c>' + shared("H6", "支付成功") + shared("I6", "12345678901234567890") + '</row>']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statement.xlsx"
            with ZipFile(path, "w") as archive:
                archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="账单" sheetId="1" r:id="rId1"/></sheets></workbook>')
                archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
                archive.writestr("xl/sharedStrings.xml", '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">' + ''.join(f'<si><t>{escape(value)}</t></si>' for value in strings) + '</sst>')
                archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + ''.join(rows) + '</sheetData></worksheet>')
            records, metadata = parse_wechat(path)
        self.assertEqual(records[0]["time"], "2026-10-22 14:05:24")
        self.assertEqual(records[0]["amount"], 30)
        self.assertEqual(records[0]["description"], "")
        self.assertEqual(records[0]["transaction_id"], "12345678901234567890")
        self.assertEqual(records[0]["row"], 6)
        self.assertTrue(metadata["totals_match"])


if __name__ == "__main__":
    unittest.main()
