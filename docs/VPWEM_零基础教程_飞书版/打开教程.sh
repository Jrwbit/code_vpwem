#!/usr/bin/env bash
set -euo pipefail

LIBRARY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${VPWEM_TUTORIAL_PORT:-8765}"
URL="http://127.0.0.1:${PORT}/index.html"

cd "$LIBRARY_DIR"
echo "VPWEM 教程地址：$URL"
echo "保持此终端运行；按 Ctrl+C 关闭教程服务。"

if command -v xdg-open >/dev/null 2>&1; then
  (sleep 1; xdg-open "$URL" >/dev/null 2>&1 || true) &
fi

exec python3 -m http.server "$PORT" --bind 127.0.0.1
