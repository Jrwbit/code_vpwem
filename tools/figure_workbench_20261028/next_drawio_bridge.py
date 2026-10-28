#!/usr/bin/env python3
"""Render editable .drawio files through the real Next AI Draw.io MCP editor.

No model API is called. Requires Node.js, Python Playwright and Chrome. The
pinned MCP package ships an offline draw.io editor. Example:

  python tools/figure_workbench_20261028/next_drawio_bridge.py \
    docs/assets/momart_main_figures_20261028/drawio/no_history.drawio \
    --output-prefix /tmp/no_history

Use --mcp-entry /path/to/@next-ai-drawio/mcp-server/dist/index.js to use an
already installed package; otherwise npx resolves the pinned version.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import shutil

from playwright.async_api import async_playwright


class StdioMCP:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self.pending: dict[int, asyncio.Future] = {}
        self.next_id = 0
        self.reader = asyncio.create_task(self.read_messages())

    async def read_messages(self):
        assert self.process.stdout is not None
        while line := await self.process.stdout.readline():
            message = json.loads(line)
            future = self.pending.pop(message.get('id'), None)
            if future is not None:
                future.set_result(message)

    async def request(self, method: str, params: dict):
        self.next_id += 1
        message_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[message_id] = future
        await self.send({'jsonrpc': '2.0', 'id': message_id,
                         'method': method, 'params': params})
        response = await asyncio.wait_for(future, timeout=120)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['result']

    async def send(self, message: dict):
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(message) + '\n').encode())
        await self.process.stdin.drain()

    async def tool(self, name: str, arguments: dict):
        result = await self.request('tools/call',
                                    {'name': name, 'arguments': arguments})
        if result.get('isError'):
            raise RuntimeError(result)
        return result


async def render(args):
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = args.output_prefix.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = (['node', str(args.mcp_entry.resolve())] if args.mcp_entry else
               ['npx', '--yes', '@next-ai-drawio/mcp-server@0.4.6'])
    env = dict(os.environ, BROWSER='none', DRAWIO_DATA_DIR='off',
               DRAWIO_LANG='en', DRAWIO_DARK='0')
    process = await asyncio.create_subprocess_exec(
        *command, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, env=env)
    client = StdioMCP(process)
    try:
        await client.request('initialize', {
            'protocolVersion': '2024-11-05', 'capabilities': {},
            'clientInfo': {'name': 'momart-figure-export', 'version': '1.0'}})
        await client.send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        session = await client.tool('start_session', {})
        text = '\n'.join(item.get('text', '') for item in session['content'])
        match = re.search(r'Browser URL: (\S+)', text)
        if not match:
            raise RuntimeError('MCP did not return a preview URL')
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=args.chrome, headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage',
                      '--disable-background-timer-throttling',
                      '--disable-renderer-backgrounding'])
            try:
                page = await browser.new_page(
                    viewport={'width': 1800, 'height': 1200},
                    device_scale_factor=2)
                await page.goto(match.group(1), wait_until='domcontentloaded')
                await page.wait_for_timeout(4000)
                await client.tool('load_diagram', {'path': str(source)})
                for suffix in args.formats:
                    path = str(output) + '.' + suffix
                    result = await client.tool('export_diagram', {'path': path})
                    for item in result['content']:
                        if item.get('type') == 'text':
                            print(item['text'], flush=True)
            finally:
                await browser.close()
    finally:
        if process.stdin is not None:
            process.stdin.close()
            await process.stdin.wait_closed()
        if process.returncode is None:
            process.terminate()
        await process.wait()
        await client.reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    parser.add_argument('--formats', nargs='+', default=['png', 'svg'],
                        choices=['png', 'svg', 'drawio.svg', 'drawio'])
    parser.add_argument('--mcp-entry', type=Path)
    parser.add_argument('--chrome', default=shutil.which('google-chrome'))
    asyncio.run(render(parser.parse_args()))


if __name__ == '__main__':
    main()
