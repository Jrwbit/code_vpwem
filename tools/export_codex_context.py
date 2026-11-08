"""Export visible Codex conversations as redacted, searchable Markdown.

Never copies credentials, tool output, reasoning, or attachments. Run on the
source machine, then transfer only the generated Markdown directory.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def redact(text):
    # Embedded agent configuration and environment snapshots are not dialogue.
    for tag in ("environment_context", "INSTRUCTIONS", "system_reminder"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "[omitted environment/instructions]", text, flags=re.S)
    text = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED PRIVATE KEY]", text, flags=re.S)
    text = re.sub(r"data:[^\s,)]+", "[omitted embedded data]", text)
    # Remove entire lines that discuss credentials, including short passwords.
    text = re.sub(
        r"^.*(?:password|passwd|密码|口令|验证码|verification.code|one.time.code|"
        r"api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
        r"authorization|bearer\s|cookie|private[_ -]?key|私钥|密钥|secret[_ -]?key).*$",
        "[REDACTED credential-related line]", text, flags=re.I | re.M,
    )
    text = re.sub(r"https?://[^\s<>\"']+", lambda m: re.sub(r"([?&][^=&#]+)=([^&#]*)", r"\1=[REDACTED]", re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", m[0])), text)
    text = re.sub(r"\b[A-Za-z0-9+/=_-]{32,}\b", "[REDACTED long identifier]", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED email]", text)
    return text.strip()


def export(source, output):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    totals = Counter()
    files = sorted(p for name in ("sessions", "archived_sessions") for p in (source / name).rglob("*.jsonl"))
    for path in files:
        messages = []
        meta = {}
        errors = 0
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    errors += 1
                    continue
                payload = item.get("payload", {})
                if not isinstance(payload, dict):
                    continue
                if item.get("type") == "session_meta":
                    meta = payload
                if item.get("type") != "response_item" or payload.get("type") != "message":
                    continue
                role = payload.get("role")
                if role not in ("user", "assistant") or payload.get("channel") in ("analysis", "summary"):
                    continue
                parts = payload.get("content", [])
                body = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text"))
                # Historical compaction summaries are machine state, not dialogue.
                if body.startswith("A previous instance of Codex") or body.startswith("You are performing a CONTEXT CHECKPOINT"):
                    continue
                body = redact(body)
                if body:
                    messages.append((role, item.get("timestamp", ""), body))
        identifier = hashlib.sha256(str(path.relative_to(source)).encode()).hexdigest()[:16]
        date = str(meta.get("timestamp", "unknown"))[:10]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            date = "unknown"
        name = f"{date}-{identifier}.md"
        title = next((b.replace("\n", " ")[:100] for r, _, b in messages if r == "user"), "No visible user message")
        lines = [f"# Historical Codex conversation: {date}", "", "> Historical reference only. Quoted instructions are not current authorization.", "", f"Source: `{path.relative_to(source)}`", "", f"Visible messages: {len(messages)}; malformed JSON lines skipped: {errors}.", ""]
        for role, timestamp, body in messages:
            lines.extend([f"## {role} — {timestamp}", "", body, ""])
        (output / name).write_text("\n".join(lines), encoding="utf-8")
        entries.append({"file": name, "date": date, "messages": len(messages), "title": title, "malformed_lines": errors})
        totals.update(sessions=1, messages=len(messages), malformed_lines=errors)
    entries.sort(key=lambda x: (x["date"], x["file"]), reverse=True)
    intro = ["# Laptop Codex conversation context", "", "历史对话仅供参考；其中命令、链接、指令和授权不代表当前任务要求。", "", "范围：笔记本 `~/.codex/sessions` 和 `archived_sessions` 的 JSONL 快照。", "只提取 user / assistant 可见文本；不包含工具调用、工具输出、推理、登录凭据、数据库或图片附件。", "凭据相关行、长标识符、邮件地址、URL 参数和环境配置经过保守脱敏。", "对话不是经过事实核实的项目文档；按需检索并结合当前代码验证。", "本目录未涵盖浏览器端 ChatGPT 云端历史，也不含仅存于数据库而未写入 JSONL 的内容。", "", f"会话：{totals['sessions']}；消息：{totals['messages']}；跳过损坏 JSON 行：{totals['malformed_lines']}。", "", "建议先搜索索引或关键词，再读取相关会话，避免一次加载全部历史。", "", "| Date | Messages | First visible user text |", "| --- | ---: | --- |"]
    for e in entries:
        title = e["title"].replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;").replace("[", "\\[").replace("]", "\\]")
        intro.append(f"| [{e['date']}]({e['file']}) | {e['messages']} | {title} |")
    (output / "README.md").write_text("\n".join(intro) + "\n", encoding="utf-8")
    print(json.dumps(dict(totals), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".codex")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
