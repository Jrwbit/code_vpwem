#!/usr/bin/env python3
"""Run Codex threads from a Feishu bot webhook.

The bridge intentionally keeps the transport small: the Feishu callback is
handled by the standard library, while Codex is driven through its local
app-server JSON-RPC protocol. Access is closed by default and requires the
configured Feishu open ID allowlist.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen


LOG = logging.getLogger("feishu-codex-bridge")
FEISHU_API = "https://open.feishu.cn/open-apis"


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


class Config:
    def __init__(self, *, require_webhook: bool = True) -> None:
        credentials_path = _env_path(
            "FEISHU_CREDENTIALS_FILE", "/home/wjr/.config/feishu-docs/credentials.json"
        )
        credentials = _load_json(credentials_path, {})
        self.feishu_app_id = os.environ.get("FEISHU_APP_ID", credentials.get("app_id", ""))
        self.feishu_app_secret = os.environ.get(
            "FEISHU_APP_SECRET", credentials.get("app_secret", "")
        )
        self.verification_token = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
        self.encrypt_key = os.environ.get("FEISHU_ENCRYPT_KEY", "")
        default_manager = credentials.get("default_manager_id", "")
        allowed = os.environ.get("FEISHU_ALLOWED_OPEN_IDS", default_manager)
        self.allowed_open_ids = {item.strip() for item in allowed.split(",") if item.strip()}
        if not self.feishu_app_id or not self.feishu_app_secret:
            raise RuntimeError("Feishu app credentials are missing")
        if require_webhook and (not self.verification_token or not self.encrypt_key):
            raise RuntimeError(
                "FEISHU_VERIFICATION_TOKEN and FEISHU_ENCRYPT_KEY are required; "
                "configure the Feishu event subscription before starting the bridge"
            )
        if not self.allowed_open_ids:
            raise RuntimeError("FEISHU_ALLOWED_OPEN_IDS is empty; refusing open access")

        self.state_path = _env_path(
            "CODEX_BRIDGE_STATE", "/home/wjr/.codex/feishu-codex-bridge/state.json"
        )
        self.codex_bin = os.environ.get("CODEX_BIN", "codex")
        self.codex_model = os.environ.get("CODEX_MODEL", "")
        self.bind = os.environ.get("CODEX_BRIDGE_BIND", "127.0.0.1")
        self.port = int(os.environ.get("CODEX_BRIDGE_PORT", "8787"))
        self.max_message_chars = int(os.environ.get("CODEX_BRIDGE_MAX_MESSAGE", "3800"))
        self.topic_chat_id = os.environ.get("FEISHU_CODEX_TOPIC_CHAT_ID", "")
        self.topic_sync_seconds = int(os.environ.get("FEISHU_CODEX_TOPIC_SYNC_SECONDS", "300"))
        self.default_manager_id = os.environ.get(
            "FEISHU_DOC_EDITOR_ID", credentials.get("default_manager_id", "")
        )
        self.default_manager_type = os.environ.get(
            "FEISHU_DOC_EDITOR_TYPE", credentials.get("default_manager_type", "openid")
        )
        self.doc_base_url = os.environ.get(
            "FEISHU_DOC_BASE_URL", "https://ccnuoe6sry36.feishu.cn/docx"
        ).rstrip("/")
        self.default_workspace = Path(
            os.environ.get("CODEX_DEFAULT_WORKSPACE", "/data/wjr/vpwem/code_vpwem")
        ).resolve()
        workspace_json = os.environ.get("CODEX_WORKSPACES_JSON", "")
        self.workspaces = {
            "vpwem": self.default_workspace,
            "motor": Path("/home/wjr/code/r435").resolve(),
        }
        if workspace_json:
            values = json.loads(workspace_json)
            self.workspaces.update({key: Path(value).resolve() for key, value in values.items()})
        self.workspaces = {
            key: path for key, path in self.workspaces.items() if path.is_dir()
        }
        if not self.workspaces:
            raise RuntimeError("No usable Codex workspace configured")


class FeishuClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._token = ""
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            body = json.dumps(
                {"app_id": self.config.feishu_app_id, "app_secret": self.config.feishu_app_secret}
            ).encode()
            payload = self._request("POST", "/auth/v3/tenant_access_token/internal", body, auth=False)
            self._token = payload["tenant_access_token"]
            self._expires_at = time.time() + int(payload.get("expire", 7200))
            return self._token

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        auth: bool = True,
        query: dict[str, str] | None = None,
        content_type: str = "application/json; charset=utf-8",
    ) -> dict[str, Any]:
        url = FEISHU_API + path
        if query:
            url += "?" + urlencode(query)
        headers = {"Content-Type": content_type}
        if auth:
            headers["Authorization"] = f"Bearer {self.token()}"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                error = json.loads(detail)
                detail = f"{error.get('code')}: {error.get('msg')}"
            except json.JSONDecodeError:
                detail = detail[:500]
            raise RuntimeError(f"Feishu HTTP {exc.code}: {detail}") from exc
        if payload.get("code", 0) != 0:
            raise RuntimeError(f"Feishu API error {payload.get('code')}: {payload.get('msg')}")
        return payload.get("data", payload)

    def reply(self, message_id: str, text: str) -> None:
        body = json.dumps({"content": json.dumps({"text": text}, ensure_ascii=False), "msg_type": "text"}).encode()
        self._request("POST", f"/im/v1/messages/{message_id}/reply", body)

    def send_text(self, chat_id: str, text: str) -> dict[str, Any]:
        body = json.dumps(
            {
                "receive_id": chat_id,
                "content": json.dumps({"text": text}, ensure_ascii=False),
                "msg_type": "text",
            }
        ).encode()
        return self._request(
            "POST",
            "/im/v1/messages",
            body,
            query={"receive_id_type": "chat_id"},
        )

    def update_text(self, message_id: str, text: str) -> None:
        body = json.dumps(
            {
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            }
        ).encode()
        self._request("PUT", f"/im/v1/messages/{message_id}", body)

    @staticmethod
    def _json_body(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def create_document(self, title: str, folder_token: str = "") -> str:
        payload = {"title": title[:800]}
        if folder_token:
            payload["folder_token"] = folder_token
        data = self._request(
            "POST",
            "/docx/v1/documents",
            self._json_body(payload),
        )
        document_id = data.get("document", {}).get("document_id", "")
        if not document_id:
            raise RuntimeError(f"飞书创建文档未返回 document_id：{data}")
        return document_id

    def convert_markdown(self, markdown: str) -> dict[str, Any]:
        data = self._request(
            "POST",
            "/docx/v1/documents/blocks/convert",
            self._json_body({"content_type": "markdown", "content": markdown}),
        )
        if not data.get("blocks") or not data.get("first_level_block_ids"):
            raise RuntimeError("飞书没有从 Markdown 转换出文档块")
        return data

    @staticmethod
    def _converted_batches(
        converted: dict[str, Any], max_top_level: int = 50
    ) -> list[tuple[list[str], list[dict[str, Any]]]]:
        """Partition converted blocks without separating parents from children."""
        blocks = converted["blocks"]
        # The converter returns table merge metadata for reading/rendering, but
        # the descendant creation API treats merge_info as read-only and rejects
        # the otherwise valid converted payload with error 1770001.
        for block in blocks:
            table_property = block.get("table", {}).get("property", {})
            table_property.pop("merge_info", None)
        by_id = {block.get("block_id"): block for block in blocks}
        batches = []
        top_level = converted["first_level_block_ids"]
        for start in range(0, len(top_level), max_top_level):
            roots = top_level[start : start + max_top_level]
            included: set[str] = set()
            pending = list(roots)
            while pending:
                block_id = pending.pop()
                if block_id in included:
                    continue
                included.add(block_id)
                pending.extend(by_id.get(block_id, {}).get("children", []))
            descendants = [
                block for block in blocks if block.get("block_id") in included
            ]
            batches.append((roots, descendants))
        return batches

    def insert_descendants(self, document_id: str, converted: dict[str, Any]) -> dict[str, str]:
        relations: dict[str, str] = {}
        for roots, descendants in self._converted_batches(converted):
            data = self._request(
                "POST",
                f"/docx/v1/documents/{document_id}/blocks/{document_id}/descendant",
                self._json_body(
                    {
                        "children_id": roots,
                        "index": -1,
                        "descendants": descendants,
                    }
                ),
                query={"document_revision_id": "-1"},
            )
            relations.update(
                {
                    item["temporary_block_id"]: item["block_id"]
                    for item in data.get("block_id_relations", [])
                    if item.get("temporary_block_id") and item.get("block_id")
                }
            )
            # Feishu limits document editing APIs to roughly three writes per
            # second. Large Markdown files need several descendant batches.
            time.sleep(0.4)
        return relations

    def add_document_editor(
        self, document_id: str, member_id: str, member_type: str = "openid"
    ) -> None:
        if not member_id:
            return
        self._request(
            "POST",
            f"/drive/v1/permissions/{document_id}/members",
            self._json_body(
                {
                    "member_type": member_type,
                    "member_id": member_id,
                    "perm": "full_access",
                }
            ),
            query={"type": "docx", "need_notification": "false"},
        )

    @staticmethod
    def _multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
        boundary = "----feishu-codex-" + uuid.uuid4().hex
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{file_path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                file_path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def upload_image(self, image_path: Path, image_block_id: str) -> str:
        body, content_type = self._multipart(
            {
                "file_name": image_path.name,
                "parent_type": "docx_image",
                "parent_node": image_block_id,
                "size": str(image_path.stat().st_size),
            },
            image_path,
        )
        data = self._request(
            "POST",
            "/drive/v1/medias/upload_all",
            body,
            content_type=content_type,
        )
        token = data.get("file_token", "")
        if not token:
            raise RuntimeError(f"飞书上传图片未返回 file_token：{data}")
        return token

    def replace_image(self, document_id: str, block_id: str, file_token: str) -> None:
        self._request(
            "PATCH",
            f"/docx/v1/documents/{document_id}/blocks/{block_id}",
            self._json_body({"replace_image": {"token": file_token}}),
            query={"document_revision_id": "-1"},
        )

    @staticmethod
    def _find_file_block(value: Any) -> str:
        if isinstance(value, dict):
            if value.get("block_type") == 23 and value.get("block_id"):
                return str(value["block_id"])
            for child in value.values():
                block_id = FeishuClient._find_file_block(child)
                if block_id:
                    return block_id
        elif isinstance(value, list):
            for child in value:
                block_id = FeishuClient._find_file_block(child)
                if block_id:
                    return block_id
        return ""

    def create_file_block(self, document_id: str) -> str:
        data = self._request(
            "POST",
            f"/docx/v1/documents/{document_id}/blocks/{document_id}/children",
            self._json_body(
                {"children": [{"block_type": 23, "file": {}}], "index": -1}
            ),
            query={"document_revision_id": "-1"},
        )
        block_id = self._find_file_block(data)
        if not block_id:
            children = data.get("children", [])
            if children and children[0].get("block_type") == 33:
                view_children = children[0].get("children", [])
                if view_children and isinstance(view_children[0], str):
                    block_id = view_children[0]
        if not block_id:
            raise RuntimeError(f"飞书创建附件块未返回内部 file block：{data}")
        return block_id

    def upload_file(self, file_path: Path, file_block_id: str) -> str:
        body, content_type = self._multipart(
            {
                "file_name": file_path.name,
                "parent_type": "docx_file",
                "parent_node": file_block_id,
                "size": str(file_path.stat().st_size),
            },
            file_path,
        )
        data = self._request(
            "POST",
            "/drive/v1/medias/upload_all",
            body,
            content_type=content_type,
        )
        token = data.get("file_token", "")
        if not token:
            raise RuntimeError(f"飞书上传附件未返回 file_token：{data}")
        return token

    def replace_file(self, document_id: str, block_id: str, file_token: str) -> None:
        self._request(
            "PATCH",
            f"/docx/v1/documents/{document_id}/blocks/{block_id}",
            self._json_body({"replace_file": {"token": file_token}}),
            query={"document_revision_id": "-1"},
        )

    def add_file_attachment(self, document_id: str, file_path: Path) -> str:
        block_id = self.create_file_block(document_id)
        token = self.upload_file(file_path, block_id)
        self.replace_file(document_id, block_id, token)
        return token

    @staticmethod
    def _local_image_path(
        image_url: str, markdown_path: Path, asset_root: Path
    ) -> Path | None:
        parsed = urlparse(image_url)
        if parsed.scheme in {"http", "https", "data"}:
            return None
        candidate = (markdown_path.parent / unquote(parsed.path)).resolve()
        try:
            candidate.relative_to(asset_root.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def publish_markdown(
        self,
        markdown_path: Path,
        title: str,
        *,
        folder_token: str = "",
        editor_id: str = "",
        editor_type: str = "openid",
        asset_root: Path | None = None,
        attachments: list[Path] | None = None,
    ) -> tuple[str, list[str]]:
        markdown = markdown_path.read_text(encoding="utf-8")
        if len(markdown) > 10_000_000:
            raise ValueError("Markdown 超过飞书文档 1000 万字符限制")
        document_id = self.create_document(title, folder_token=folder_token)
        converted = self.convert_markdown(markdown)
        relations = self.insert_descendants(document_id, converted)
        warnings: list[str] = []
        asset_root = (asset_root or markdown_path.parent).resolve()
        for image in converted.get("block_id_to_image_urls", []):
            temporary_id = image.get("block_id", "")
            block_id = relations.get(temporary_id, "")
            image_path = self._local_image_path(
                image.get("image_url", ""), markdown_path, asset_root
            )
            if not block_id or image_path is None:
                warnings.append(f"未上传图片：{image.get('image_url', '<unknown>')}")
                continue
            try:
                token = self.upload_image(image_path, block_id)
                self.replace_image(document_id, block_id, token)
                time.sleep(0.4)
            except Exception as exc:
                warnings.append(f"图片 {image_path.name} 上传失败：{exc}")
        for attachment in attachments or []:
            try:
                self.add_file_attachment(document_id, attachment)
                time.sleep(0.4)
            except Exception as exc:
                warnings.append(f"附件 {attachment.name} 上传失败：{exc}")
        self.add_document_editor(document_id, editor_id, editor_type)
        return f"{self.config.doc_base_url}/{document_id}", warnings


class CodexAppServer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None
        self.responses: dict[int, queue.Queue[dict[str, Any]]] = {}
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.response_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.turn_lock = threading.RLock()
        self.next_id = 1
        self.turn_ids: dict[str, str] = {}
        self.start()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [self.config.codex_bin, "app-server", "--listen", "stdio://"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_loop, name="codex-app-server-reader", daemon=True)
        self.reader.start()
        self._request("initialize", {"clientInfo": {"name": "feishu_codex_bridge", "title": "Feishu Codex Bridge", "version": "0.1.0"}})
        self._notify("initialized", {})

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.reader = None

    def restart(self) -> None:
        self.close()
        self.events = queue.Queue()
        self.turn_ids.clear()
        self.start()

    def _read_loop(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("Ignoring non-JSON app-server output")
                continue
            request_id = message.get("id")
            if isinstance(request_id, int):
                with self.response_lock:
                    waiter = self.responses.get(request_id)
                if waiter:
                    waiter.put(message)
            elif message.get("method"):
                self.events.put(message)

    def _send(self, message: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("Codex app-server is not running")
        with self.write_lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def _request(self, method: str, params: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        waiter: queue.Queue[dict[str, Any]] = queue.Queue()
        with self.response_lock:
            self.responses[request_id] = waiter
        try:
            self._send({"id": request_id, "method": method, "params": params})
            response = waiter.get(timeout=timeout)
            if "error" in response:
                raise RuntimeError(f"Codex app-server error: {response['error']}")
            return response.get("result", {})
        finally:
            with self.response_lock:
                self.responses.pop(request_id, None)

    def new_thread(self, workspace: Path) -> str:
        with self.turn_lock:
            params: dict[str, Any] = {
                "cwd": str(workspace),
                "sandbox": "danger-full-access",
                "approvalPolicy": "never",
                "developerInstructions": (
                    "You are controlled by the owner through a private Feishu bot. "
                    "Work directly in the requested repository, report concrete results, "
                    "and do not wait for interactive approval."
                ),
            }
            if self.config.codex_model:
                params["model"] = self.config.codex_model
            result = self._request("thread/start", params)
            thread = result.get("thread", {})
            thread_id = thread.get("id")
            if not thread_id:
                raise RuntimeError(f"Codex did not return a thread id: {result}")
            return thread_id

    def set_thread_name(self, thread_id: str, name: str) -> None:
        with self.turn_lock:
            self._request("thread/name/set", {"threadId": thread_id, "name": name})

    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.turn_lock:
            result = self._request(
                "thread/list",
                {
                    "limit": limit,
                    "sortKey": "updated_at",
                    "sortDirection": "desc",
                    "sourceKinds": ["cli", "vscode", "appServer"],
                },
                timeout=60,
            )
            return result.get("data", [])

    def read_thread(self, thread_id: str) -> dict[str, Any]:
        """Read one native thread by ID, including threads outside list pagination."""
        with self.turn_lock:
            result = self._request(
                "thread/read",
                {"threadId": thread_id, "includeTurns": False},
                timeout=60,
            )
            thread = result.get("thread", {})
            if not thread.get("id"):
                raise RuntimeError(f"Codex did not return thread metadata: {result}")
            return thread

    def resume_thread(self, thread_id: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "sandbox": "danger-full-access",
            "approvalPolicy": "never",
            "developerInstructions": (
                "You are controlled by the owner through VSCode and a private Feishu bot. "
                "Continue the same shared Codex thread, work directly in the requested repository, "
                "report concrete results, and do not wait for interactive approval."
            ),
        }
        if self.config.codex_model:
            params["model"] = self.config.codex_model
        with self.turn_lock:
            return self._request("thread/resume", params, timeout=60)

    def archive_thread(self, thread_id: str) -> None:
        with self.turn_lock:
            self._request("thread/archive", {"threadId": thread_id})

    def run(self, thread_id: str, prompt: str) -> str:
        with self.turn_lock:
            # The IDE uses another app-server process. Reload from disk before
            # each Feishu turn so any turns added in VSCode are included.
            self.restart()
            self.resume_thread(thread_id)
            return self._run_locked(thread_id, prompt)

    def _run_locked(self, thread_id: str, prompt: str) -> str:
        result = self._request(
            "turn/start",
            {"threadId": thread_id, "input": [{"type": "text", "text": prompt}]},
        )
        turn = result.get("turn", {})
        turn_id = turn.get("id")
        if turn_id:
            self.turn_ids[thread_id] = turn_id
        chunks: list[str] = []
        deadline = time.time() + 3600
        while time.time() < deadline:
            message = self._next_event(timeout=5)
            if not message:
                continue
            method = message.get("method", "")
            params = message.get("params", {})
            if method == "item/agentMessage/delta":
                delta = params.get("delta", "")
                if delta:
                    chunks.append(delta)
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage" and item.get("text"):
                    if not chunks:
                        chunks.append(item["text"])
            elif method == "turn/completed":
                self.turn_ids.pop(thread_id, None)
                break
            elif method == "turn/failed":
                self.turn_ids.pop(thread_id, None)
                raise RuntimeError(str(params.get("error", "Codex turn failed")))
        else:
            raise TimeoutError("Codex turn exceeded one hour")
        return "".join(chunks).strip() or "Codex completed without a final message."

    def _next_event(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def interrupt(self, thread_id: str) -> None:
        turn_id = self.turn_ids.get(thread_id)
        if turn_id:
            self._request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})


class BridgeState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data = _load_json(path, {"chats": {}, "events": {}})
        self.data.setdefault("chats", {})
        self.data.setdefault("events", {})
        self.data.setdefault("topics", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)


class Bridge:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.feishu = FeishuClient(config)
        self.codex = CodexAppServer(config)
        self.state = BridgeState(config.state_path)
        self.state_lock = threading.Lock()
        self.topic_sync_lock = threading.Lock()
        if self.config.topic_chat_id:
            threading.Thread(target=self._topic_sync_loop, name="codex-topic-sync", daemon=True).start()

    def handle_event(self, payload: dict[str, Any]) -> None:
        event_id = payload.get("header", {}).get("event_id", "")
        with self.state.lock:
            if event_id and event_id in self.state.data["events"]:
                return
            if event_id:
                self.state.data["events"][event_id] = int(time.time())
                self.state.data["events"] = {
                    key: value for key, value in self.state.data["events"].items() if value > time.time() - 86400
                }
                self.state.save()
        event = payload.get("event", {})
        sender_id = event.get("sender", {}).get("sender_id", {})
        open_id = sender_id.get("open_id", "")
        LOG.info("handling Feishu event sender=%s", open_id or "<missing>")
        if open_id not in self.config.allowed_open_ids:
            LOG.warning("Rejected Feishu sender %s", open_id or "<missing>")
            return
        message = event.get("message", {})
        message_id = message.get("message_id", "")
        chat_id = message.get("chat_id", "")
        if not message_id or not chat_id:
            return
        raw_content = message.get("content", "")
        try:
            content = json.loads(raw_content).get("text", "")
        except (TypeError, json.JSONDecodeError):
            content = raw_content
        content = content.strip()
        if not content:
            return
        threading.Thread(
            target=self.process_message,
            args=(
                chat_id,
                message_id,
                content,
                message.get("thread_id", ""),
                message.get("root_id", ""),
                open_id,
            ),
            daemon=True,
        ).start()

    def process_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        feishu_thread_id: str = "",
        root_id: str = "",
        sender_open_id: str = "",
    ) -> None:
        try:
            # Group mentions are included in the text payload (for example
            # "@_user_1 /status" or an <at> tag). Commands must be recognized
            # after removing that transport-specific prefix.
            content = self._strip_bot_mention(content).strip()
            if not content:
                self.feishu.reply(message_id, "已收到 @，请附带命令或任务内容。发送 /help 查看用法。")
                return
            command = content.split()
            action = command[0].lower()
            if action in {"/publish", "/doc", "发布文档"}:
                args = shlex.split(content)
                if len(args) < 2:
                    raise ValueError(
                        "请指定工作区内的 Markdown，例如："
                        "/publish docs/report.md"
                    )
                with self.state.lock:
                    workspace_name = self._chat_state(chat_id)["workspace"]
                markdown_path = self._resolve_workspace_file(
                    self._workspace(workspace_name), args[1]
                )
                title = " ".join(args[2:]).strip() or self._markdown_title(markdown_path)
                self.feishu.reply(message_id, f"正在创建飞书文档：{title}")
                editor_id = sender_open_id or self.config.default_manager_id
                editor_type = "openid" if sender_open_id else self.config.default_manager_type
                url, warnings = self.feishu.publish_markdown(
                    markdown_path,
                    title,
                    editor_id=editor_id,
                    editor_type=editor_type,
                    asset_root=self._workspace(workspace_name),
                )
                result = f"飞书文档已创建并授予你编辑权限：\n{url}"
                if warnings:
                    result += "\n\n注意：\n" + "\n".join(warnings[:10])
                self.feishu.reply(message_id, result)
                return
            if action in {"/new", "新对话"}:
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    args = command[1:]
                    workspace_name = chat["workspace"]
                    if args and args[0] in self.config.workspaces:
                        workspace_name = args.pop(0)
                thread_name = " ".join(args)
                thread_id = self.codex.new_thread(self._workspace(workspace_name))
                if thread_name:
                    self.codex.set_thread_name(thread_id, thread_name)
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    chat["workspace"] = workspace_name
                    chat["active_thread_id"] = thread_id
                    self.state.save()
                self.feishu.reply(
                    message_id,
                    f"已创建 Codex 原生对话：{thread_name or thread_id[:8]}\n"
                    f"工作区：{workspace_name}\nVSCode 与飞书共用此记录。",
                )
                return
            if action == "/use":
                workspace_name = command[1] if len(command) > 1 else "vpwem"
                thread_id = self.codex.new_thread(self._workspace(workspace_name))
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    chat["workspace"] = workspace_name
                    chat["active_thread_id"] = thread_id
                    self.state.save()
                self.feishu.reply(
                    message_id,
                    f"已切换到工作区 {workspace_name}，并创建 Codex 原生对话 {thread_id[:8]}",
                )
                return
            if action in {"/sync", "/topics", "同步对话"}:
                created, updated = self.sync_topics()
                self.feishu.reply(
                    message_id,
                    f"Codex 对话目录已同步：新增 {created} 个话题，更新 {updated} 个标题。",
                )
                return
            if action in {"/sessions", "/list", "对话列表"}:
                workspace_filter = command[1] if len(command) > 1 else ""
                if workspace_filter and workspace_filter not in self.config.workspaces:
                    raise ValueError("筛选工作区只能是 vpwem 或 motor")
                threads = self._visible_threads(workspace_filter, limit=20)
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    active_id = chat.get("active_thread_id", "")
                    chat["last_threads"] = [thread["id"] for thread in threads]
                    self.state.save()
                lines = []
                for index, thread in enumerate(threads, 1):
                    marker = "*" if thread["id"] == active_id else " "
                    title = thread.get("name") or thread.get("preview") or "未命名对话"
                    title = " ".join(title.split())[:34]
                    workspace_name = self._workspace_name(thread.get("cwd", ""))
                    updated = datetime.fromtimestamp(thread["updatedAt"]).strftime("%m-%d %H:%M")
                    lines.append(f"{marker}{index}. [{workspace_name}] {title} ({updated})")
                self.feishu.reply(
                    message_id,
                    "Codex/VSCode 对话（* 为飞书当前）：\n"
                    + ("\n".join(lines) if lines else "暂无对话，请使用 /new")
                    + "\n发送 /switch 编号 选择。",
                )
                return
            if action in {"/switch", "/bind", "/chat", "切换对话", "绑定对话"}:
                if len(command) < 2:
                    raise ValueError("请指定编号、标题或 thread ID，例如：/bind 019f...")
                selector = " ".join(command[1:])
                thread = self._select_thread(chat_id, selector)
                workspace_name = self._workspace_name(thread.get("cwd", ""))
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    chat["active_thread_id"] = thread["id"]
                    if workspace_name in self.config.workspaces:
                        chat["workspace"] = workspace_name
                    self.state.save()
                title = thread.get("name") or thread.get("preview") or thread["id"][:8]
                self.feishu.reply(
                    message_id,
                    f"已绑定同一个 Codex thread：{title}\nthread ID：{thread['id']}\n"
                    f"工作区：{workspace_name}\n"
                    "接下来的飞书消息会写入该记录。",
                )
                return
            if action in {"/archive", "/delete", "删除对话"}:
                if len(command) < 2:
                    raise ValueError("请指定编号、标题或 thread ID，例如：/archive 1")
                thread = self._select_thread(chat_id, " ".join(command[1:]))
                self.codex.archive_thread(thread["id"])
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    if chat.get("active_thread_id") == thread["id"]:
                        chat["active_thread_id"] = ""
                    self.state.save()
                self.feishu.reply(message_id, "已归档该 Codex 对话，VSCode 对话列表也会同步变化。")
                return
            if action == "/stop":
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    thread_id = chat.get("active_thread_id")
                if thread_id:
                    self.codex.interrupt(thread_id)
                self.feishu.reply(message_id, "已请求停止当前 Codex 任务。")
                return
            if action == "/status":
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    thread_id = chat.get("active_thread_id", "")
                self.feishu.reply(
                    message_id,
                    f"工作区：{chat['workspace']}\n当前 Codex thread：{thread_id or '未选择'}\n"
                    "记录源：~/.codex（VSCode/飞书共享）\n完整访问：已启用",
                )
                return
            if action == "/help":
                self.feishu.reply(
                    message_id,
                    "/sessions [vpwem|motor] 查看 VSCode/Codex 对话\n"
                    "/bind thread_id 绑定 VSCode 当前对话\n"
                    "/switch 编号|标题|thread_id 选择同一个对话\n"
                    "/new [vpwem|motor] [名称] 新建共享对话\n"
                    "/sync 同步到“Codex 对话”话题群\n"
                    "/archive 编号 归档共享对话\n/use vpwem|motor 切换工作区并新建对话\n"
                    "/publish 相对路径.md [标题] 发布为可编辑飞书文档\n"
                    "/status 查看当前状态\n/stop 停止当前任务",
                )
                return
            topic = self._topic_for_message(feishu_thread_id, root_id)
            if topic:
                thread_id = topic["codex_thread_id"]
                workspace_name = topic.get("workspace", "vpwem")
            else:
                with self.state.lock:
                    chat = self._chat_state(chat_id)
                    workspace_name = chat["workspace"]
                    thread_id = chat.get("active_thread_id", "")
            if chat_id == self.config.topic_chat_id and not topic:
                thread_id = self.codex.new_thread(self._workspace(workspace_name))
                thread_name = " ".join(content.split())[:80]
                if thread_name:
                    self.codex.set_thread_name(thread_id, thread_name)
                self._bind_topic(
                    thread_id,
                    feishu_thread_id,
                    root_id or message_id,
                    root_id or message_id,
                    thread_name,
                    workspace_name,
                )
            if not thread_id:
                self.feishu.reply(
                    message_id,
                    "当前飞书会话尚未绑定 Codex thread。请在 VSCode 对话中取得 thread ID，"
                    "发送 /bind <thread_id>；或发送 /sessions 后用 /switch 编号绑定。"
                    "如需独立新对话，请显式发送 /new。",
                )
                return
            self.feishu.reply(
                message_id,
                f"已收到，正在共享 Codex 对话 {thread_id[:8]} 中执行（{workspace_name}，完整访问）。",
            )
            result = self.codex.run(thread_id, content)
            self._reply_chunks(message_id, result)
        except Exception as exc:
            LOG.exception("Failed to process Feishu message")
            try:
                self.feishu.reply(message_id, f"Codex 执行失败：{exc}")
            except Exception:
                LOG.exception("Failed to send Feishu error reply")

    @staticmethod
    def _strip_bot_mention(content: str) -> str:
        content = re.sub(r"<at[^>]*>.*?</at>\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"^@(?:_user_\d+|电机检测)\s*", "", content, flags=re.IGNORECASE)
        return content

    @staticmethod
    def _resolve_workspace_file(workspace: Path, value: str) -> Path:
        workspace = workspace.resolve()
        candidate = (workspace / value).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("只允许发布当前工作区内的 Markdown 文件") from exc
        if candidate.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("只支持发布 Markdown 文件（.md 或 .markdown）")
        if not candidate.is_file():
            raise ValueError(f"文件不存在：{value}")
        return candidate

    @staticmethod
    def _markdown_title(path: Path) -> str:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                match = re.match(r"^#\s+(.+?)\s*$", line)
                if match:
                    return re.sub(r"[`*_]", "", match.group(1))[:800]
        return path.stem[:800]

    def _workspace(self, name: str) -> Path:
        if name not in self.config.workspaces:
            raise ValueError(f"未知工作区 {name}，可选：{', '.join(self.config.workspaces)}")
        return self.config.workspaces[name]

    def _workspace_name(self, cwd: str) -> str:
        try:
            resolved = Path(cwd).resolve()
        except (OSError, RuntimeError):
            return "other"
        for name, path in self.config.workspaces.items():
            if resolved == path.resolve():
                return name
        return "other"

    def _topic_title(self, thread: dict[str, Any]) -> tuple[str, str]:
        workspace_name = self._workspace_name(thread.get("cwd", ""))
        labels = {"vpwem": "VPWEM", "motor": "电机检测", "other": "其他"}
        title = thread.get("name") or thread.get("preview") or thread["id"][:8]
        title = " ".join(title.split())[:80]
        return f"{labels[workspace_name]} | {title}", workspace_name

    def _bind_topic(
        self,
        codex_thread_id: str,
        feishu_thread_id: str,
        root_id: str,
        message_id: str,
        title: str,
        workspace: str,
    ) -> None:
        with self.state.lock:
            self.state.data["topics"][codex_thread_id] = {
                "feishu_thread_id": feishu_thread_id,
                "root_id": root_id,
                "message_id": message_id,
                "title": title,
                "workspace": workspace,
            }
            self.state.save()

    def _topic_for_message(self, feishu_thread_id: str, root_id: str) -> dict[str, Any] | None:
        if not feishu_thread_id and not root_id:
            return None
        with self.state.lock:
            for codex_thread_id, topic in self.state.data["topics"].items():
                if (
                    feishu_thread_id
                    and topic.get("feishu_thread_id") == feishu_thread_id
                ) or (
                    root_id
                    and root_id in {topic.get("root_id"), topic.get("message_id")}
                ):
                    return {"codex_thread_id": codex_thread_id, **topic}
        return None

    def sync_topics(self) -> tuple[int, int]:
        if not self.config.topic_chat_id:
            raise RuntimeError("尚未配置 Codex 话题群")
        created = 0
        updated = 0
        with self.topic_sync_lock:
            threads = self.codex.list_threads(limit=100)
            with self.state.lock:
                known = dict(self.state.data["topics"])
            for thread in reversed(threads):
                title, workspace_name = self._topic_title(thread)
                topic = known.get(thread["id"])
                if not topic:
                    sent = self.feishu.send_text(self.config.topic_chat_id, title)
                    message_id = sent.get("message_id", "")
                    feishu_thread_id = sent.get("thread_id", "")
                    if not message_id or not feishu_thread_id:
                        raise RuntimeError(f"飞书未返回话题 ID：{sent}")
                    self._bind_topic(
                        thread["id"],
                        feishu_thread_id,
                        message_id,
                        message_id,
                        title,
                        workspace_name,
                    )
                    created += 1
                elif topic.get("title") != title:
                    self.feishu.update_text(topic["message_id"], title)
                    self._bind_topic(
                        thread["id"],
                        topic.get("feishu_thread_id", ""),
                        topic.get("root_id", topic["message_id"]),
                        topic["message_id"],
                        title,
                        workspace_name,
                    )
                    updated += 1
        return created, updated

    def _topic_sync_loop(self) -> None:
        while True:
            try:
                self.sync_topics()
            except Exception:
                LOG.exception("Failed to synchronize Codex topics")
            time.sleep(self.config.topic_sync_seconds)

    def _visible_threads(self, workspace_filter: str = "", limit: int = 20) -> list[dict[str, Any]]:
        threads = self.codex.list_threads(limit=100)
        if workspace_filter:
            threads = [
                thread
                for thread in threads
                if self._workspace_name(thread.get("cwd", "")) == workspace_filter
            ]
        return threads[:limit]

    def _select_thread(self, chat_id: str, selector: str) -> dict[str, Any]:
        selector = selector.strip()
        cached_id = ""
        if selector.isdigit():
            with self.state.lock:
                chat = self._chat_state(chat_id)
                last_threads = chat.get("last_threads", [])
            index = int(selector) - 1
            if index < 0 or index >= len(last_threads):
                raise ValueError("编号无效，请先发送 /sessions 获取最新列表")
            cached_id = last_threads[index]

        # Full native IDs should be resolved directly. thread/list is paginated,
        # so using it to validate an ID made older VSCode sessions impossible to bind.
        if self._looks_like_full_thread_id(selector):
            try:
                return self.codex.read_thread(selector)
            except RuntimeError as exc:
                raise ValueError(f"未找到 Codex thread：{selector}") from exc

        threads = self.codex.list_threads(limit=100)
        if cached_id:
            match = next((thread for thread in threads if thread["id"] == cached_id), None)
            if match:
                return match
            raise ValueError("该编号对应的对话已不存在，请重新发送 /sessions")

        exact = [
            thread
            for thread in threads
            if thread["id"] == selector
            or thread["id"].startswith(selector)
            or thread.get("name") == selector
        ]
        if len(exact) == 1:
            return exact[0]
        partial = [
            thread
            for thread in threads
            if selector in (thread.get("name") or "")
            or selector in (thread.get("preview") or "")
        ]
        if len(partial) == 1:
            return partial[0]
        if len(exact) > 1 or len(partial) > 1:
            raise ValueError("匹配到多个对话，请使用 /sessions 后按编号选择")
        raise ValueError(f"未找到 Codex 对话：{selector}")

    @staticmethod
    def _looks_like_full_thread_id(value: str) -> bool:
        import re

        return bool(
            re.fullmatch(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                value,
            )
        )

    def _chat_state(self, chat_id: str) -> dict[str, Any]:
        chat = self.state.data["chats"].setdefault(chat_id, {})
        workspace = chat.get("workspace", "vpwem")
        if workspace not in self.config.workspaces:
            workspace = next(
                (
                    name
                    for name, path in self.config.workspaces.items()
                    if path.name == workspace
                ),
                "vpwem",
            )
        active_thread_id = chat.get("active_thread_id") or chat.get("thread_id", "")
        if not active_thread_id and chat.get("active"):
            active_thread_id = chat.get("sessions", {}).get(chat["active"], {}).get("thread_id", "")
        last_threads = chat.get("last_threads", [])
        chat.clear()
        chat.update(
            {
                "workspace": workspace,
                "active_thread_id": active_thread_id,
                "last_threads": last_threads,
            }
        )
        return chat

    def _reply_chunks(self, message_id: str, text: str) -> None:
        limit = self.config.max_message_chars
        for start in range(0, len(text), limit):
            self.feishu.reply(message_id, text[start : start + limit])


def _signature_valid(config: Config, headers: dict[str, str], body: bytes) -> bool:
    if not config.encrypt_key:
        return True
    timestamp = headers.get("x-lark-request-timestamp", "")
    nonce = headers.get("x-lark-request-nonce", "")
    signature = headers.get("x-lark-signature", "")
    expected = hashlib.sha256((timestamp + nonce + config.encrypt_key).encode() + body).hexdigest()
    return bool(signature) and hmac.compare_digest(signature, expected)


class WebhookHandler(BaseHTTPRequestHandler):
    bridge: Bridge

    def log_message(self, format: str, *args: Any) -> None:
        LOG.info("webhook %s", format % args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/webhook/feishu":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if not _signature_valid(self.bridge.config, {key.lower(): value for key, value in self.headers.items()}, body):
            self._json(403, {"error": "invalid signature"})
            return
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if payload.get("type") == "url_verification":
            if self.bridge.config.verification_token and payload.get("token") != self.bridge.config.verification_token:
                self._json(403, {"error": "invalid verification token"})
            else:
                self._json(200, {"challenge": payload.get("challenge", "")})
            return
        self._json(200, {"code": 0})
        self.bridge.handle_event(payload)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    config = Config()
    bridge = Bridge(config)
    WebhookHandler.bridge = bridge
    server = ThreadingHTTPServer((config.bind, config.port), WebhookHandler)
    LOG.info("Feishu Codex bridge listening on %s:%s", config.bind, config.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
