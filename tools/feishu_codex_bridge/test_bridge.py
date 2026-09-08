import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from bridge import Bridge, FeishuClient


class ThreadSelectionTests(unittest.TestCase):
    def test_full_thread_id_reads_outside_list_page(self) -> None:
        bridge = Bridge.__new__(Bridge)
        bridge.codex = SimpleNamespace(
            read_thread=Mock(return_value={"id": "019f88d5-b0d9-7151-9ea2-f5363eba9ea5"}),
            list_threads=Mock(side_effect=AssertionError("list should not be used")),
        )
        thread_id = "019f88d5-b0d9-7151-9ea2-f5363eba9ea5"
        self.assertEqual(bridge._select_thread("chat", thread_id)["id"], thread_id)
        bridge.codex.read_thread.assert_called_once_with(thread_id)

    def test_thread_id_shape_validation(self) -> None:
        self.assertTrue(Bridge._looks_like_full_thread_id("019f88d5-b0d9-7151-9ea2-f5363eba9ea5"))
        self.assertFalse(Bridge._looks_like_full_thread_id("019f88d5"))
        self.assertFalse(Bridge._looks_like_full_thread_id("not-a-thread"))


class DocumentPublishingTests(unittest.TestCase):
    def test_workspace_file_rejects_escape(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            report = workspace / "report.md"
            report.write_text("# Report\n", encoding="utf-8")
            self.assertEqual(
                Bridge._resolve_workspace_file(workspace, "report.md"), report
            )
            with self.assertRaisesRegex(ValueError, "工作区内"):
                Bridge._resolve_workspace_file(workspace, "../secret.md")

    def test_markdown_title_uses_first_h1(self) -> None:
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            report.write_text("preface\n# **VPWEM** Report\n", encoding="utf-8")
            self.assertEqual(Bridge._markdown_title(report), "VPWEM Report")

    def test_document_api_payloads(self) -> None:
        client = FeishuClient(SimpleNamespace(doc_base_url="https://example.feishu.cn/docx"))
        calls = []

        def request(method, path, body=None, **kwargs):
            calls.append((method, path, body, kwargs))
            if path == "/docx/v1/documents":
                return {"document": {"document_id": "doc123"}}
            if path.endswith("/descendant"):
                return {
                    "block_id_relations": [
                        {"temporary_block_id": "tmp1", "block_id": "real1"}
                    ]
                }
            return {}

        client._request = request
        self.assertEqual(client.create_document("Title"), "doc123")
        relations = client.insert_descendants(
            "doc123",
            {
                "first_level_block_ids": ["tmp1"],
                "blocks": [{"block_id": "tmp1", "block_type": 2}],
            },
        )
        self.assertEqual(relations, {"tmp1": "real1"})
        client.add_document_editor("doc123", "ou_editor", "openid")
        self.assertEqual(calls[0][1], "/docx/v1/documents")
        self.assertEqual(calls[1][1], "/docx/v1/documents/doc123/blocks/doc123/descendant")
        self.assertEqual(calls[2][1], "/drive/v1/permissions/doc123/members")
        self.assertIn(b'"perm": "full_access"', calls[2][2])

    def test_converted_batches_keep_nested_blocks_together(self) -> None:
        converted = {
            "first_level_block_ids": ["root1", "root2"],
            "blocks": [
                {"block_id": "root1", "children": ["child1"]},
                {"block_id": "child1", "children": []},
                {"block_id": "root2", "children": []},
            ],
        }
        batches = FeishuClient._converted_batches(converted, max_top_level=1)
        self.assertEqual(batches[0][0], ["root1"])
        self.assertEqual(
            {block["block_id"] for block in batches[0][1]}, {"root1", "child1"}
        )
        self.assertEqual(
            {block["block_id"] for block in batches[1][1]}, {"root2"}
        )

    def test_converted_batches_strip_read_only_table_merge_info(self) -> None:
        converted = {
            "first_level_block_ids": ["table1"],
            "blocks": [
                {
                    "block_id": "table1",
                    "block_type": 31,
                    "table": {"property": {"row_size": 1, "merge_info": []}},
                }
            ],
        }
        batches = FeishuClient._converted_batches(converted)
        table_property = batches[0][1][0]["table"]["property"]
        self.assertNotIn("merge_info", table_property)


if __name__ == "__main__":
    unittest.main()
