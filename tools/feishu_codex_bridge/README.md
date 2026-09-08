# Feishu Codex Bridge

This service receives messages from a Feishu bot and runs them in the same
persistent Codex threads used by the CLI and VSCode extension. It is
intentionally restricted to the configured Feishu Open ID allowlist.

## Capabilities

- `/sessions [vpwem|motor]` lists the native Codex/VSCode conversation history.
- `/bind thread-id` binds Feishu directly to the current VSCode Codex thread.
- `/switch number|title|thread-id` selects the same native thread from Feishu.
- `/new [vpwem|motor] [name]` starts a shared native conversation.
- `/archive number|title|thread-id` archives it in the shared Codex history.
- `/use vpwem` or `/use motor` switches the workspace and starts a new conversation.
- `/stop` interrupts the active turn.
- `/publish path/to/file.md [title]` creates a Feishu document from a Markdown
  file inside the active workspace, grants the sender edit access, uploads
  local images referenced by the Markdown, and replies with the document URL.
- Normal messages continue the bound thread and return the final response. When
  no thread is bound, the bridge asks for `/bind`, `/switch`, or `/new` instead
  of silently creating a separate conversation.
- Codex runs with `danger-full-access` and `approvalPolicy=never`, as explicitly
  requested. This allows arbitrary file and command changes on the host. Keep
  the Feishu app private and keep the allowlist limited to your own Open ID.

## Configuration

Copy `.env.example` into the service environment and fill in the Feishu event
verification values from the Feishu developer console. The existing credentials
file is used by default for the app ID, app secret, and default manager Open ID.
Configure the event subscription to send the normal (not AES-encrypted) JSON
payload while keeping signature verification enabled; this bridge validates the
request signature using `FEISHU_ENCRYPT_KEY`.

```bash
python3 tools/feishu_codex_bridge/bridge.py
```

For a workstation, long-connection mode avoids a public callback URL:

```bash
python3 -m pip install -r tools/feishu_codex_bridge/requirements.txt
python3 tools/feishu_codex_bridge/long_connection.py
```

Markdown can also be published without running the message bridge:

```bash
python3 tools/feishu_codex_bridge/publish_markdown.py docs/report.md
```

To create the document directly inside a Feishu Drive folder, pass its token:

```bash
python3 tools/feishu_codex_bridge/publish_markdown.py docs/report.md \
  --folder-token FOLDER_TOKEN
```

The Feishu app needs permissions to create/edit Docx documents, upload media,
and add document collaborators. Markdown conversion uses Feishu's official
Docx conversion endpoint. Only files inside the selected workspace can be
published through the bot command.

In the Feishu developer console, enable long-connection event delivery and
subscribe to the private-message/group-message receive event. The long-
connection entry point does not need `FEISHU_VERIFICATION_TOKEN` or
`FEISHU_ENCRYPT_KEY`; the HTTP entry point requires both.

Expose `/webhook/feishu` through an HTTPS reverse proxy and configure the Feishu
event subscription URL to that path. Enable the bot message-received event.
`/healthz` is available for a local health check.

The Feishu event server and the Codex bridge must run on a host that remains
online. The app-server protocol and thread lifecycle are versioned with the
installed Codex CLI; update this bridge's protocol client if the CLI changes.
