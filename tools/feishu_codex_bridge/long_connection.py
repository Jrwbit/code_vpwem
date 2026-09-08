#!/usr/bin/env python3
"""Run the Feishu Codex bridge through Feishu's long-connection mode.

Long-connection mode is useful on a workstation because it does not require a
public HTTPS callback URL. Install the optional ``lark-oapi`` dependency and
enable long connection event delivery for the bot in the Feishu developer
console.
"""

from __future__ import annotations

import json
import logging

import lark_oapi as lark

try:
    from bridge import Bridge, Config
except ImportError:  # pragma: no cover - supports `python -m` invocation.
    from .bridge import Bridge, Config


LOG = logging.getLogger("feishu-codex-bridge")


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    config = Config(require_webhook=False)
    bridge = Bridge(config)

    def on_message(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        payload = json.loads(lark.JSON.marshal(data))
        event = payload.get("event", {})
        message = event.get("message", {})
        LOG.info("received Feishu message event message_id=%s chat_id=%s", message.get("message_id"), message.get("chat_id"))
        bridge.handle_event(payload)

    def on_message_read(_: lark.im.v1.P2ImMessageMessageReadV1) -> None:
        return

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(on_message)
        .register_p2_im_message_message_read_v1(on_message_read)
        .build()
    )
    client = lark.ws.Client(
        config.feishu_app_id,
        config.feishu_app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )
    client.start()


if __name__ == "__main__":
    main()
