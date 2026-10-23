#!/usr/bin/env python3
"""Append a local MP4 to the existing report and verify its downloaded bytes.

The receipt is written before mutations and prevents accidental duplicate
appends. A failed upload is never retried automatically; --verify-only is read
only. Credentials and access tokens are neither printed nor saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feishu_codex_bridge.bridge import Config, FEISHU_API, FeishuClient


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def root_ids(client: FeishuClient, document_id: str) -> list[str]:
    result = []
    page_token = None
    while True:
        query = {'page_size': '500', 'document_revision_id': '-1'}
        if page_token:
            query['page_token'] = page_token
        data = client._request(
            'GET', f'/docx/v1/documents/{document_id}/blocks/{document_id}/children',
            query=query)
        result.extend(block['block_id'] for block in data.get('items', []))
        if not data.get('has_more'):
            return result
        next_token = data.get('page_token')
        if not next_token or next_token == page_token:
            raise RuntimeError('Invalid root pagination')
        page_token = next_token


def get_block(client: FeishuClient, document_id: str, block_id: str) -> dict:
    return client._request(
        'GET', f'/docx/v1/documents/{document_id}/blocks/{block_id}',
        query={'document_revision_id': '-1'})['block']


class MediaRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def download_digest(client: FeishuClient, file_token: str) -> tuple[int, str]:
    request = Request(f'{FEISHU_API}/drive/v1/medias/{file_token}/download',
                      headers={'Authorization': f'Bearer {client.token()}'})
    digest = hashlib.sha256()
    size = 0
    try:
        with build_opener(MediaRedirect()).open(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f'Unexpected media HTTP status: {response.status}')
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except HTTPError as exc:
        # Do not put response bodies or signed download URLs into the receipt.
        raise RuntimeError(f'Media download failed with HTTP {exc.code}') from None
    return size, digest.hexdigest()


def verify(client: FeishuClient, receipt: dict) -> None:
    doc = receipt['document_id']
    before = receipt['original_root_ids']
    after = root_ids(client, doc)
    if after[:len(before)] != before:
        raise RuntimeError('Original document root prefix changed')
    for item in receipt['text_blocks']:
        block = get_block(client, doc, item['block_id'])
        field = item['field']
        actual = ''.join(element.get('text_run', {}).get('content', '')
                         for element in block.get(field, {}).get('elements', []))
        if actual != item['text']:
            raise RuntimeError(f'Text readback mismatch: {field}')
        if item['block_id'] not in after[len(before):]:
            raise RuntimeError('Appended text root missing')
    file_block = get_block(client, doc, receipt['file_block_id'])
    uploaded = file_block.get('file', {})
    if file_block.get('block_type') != 23 or uploaded.get('token') != receipt['file_token']:
        raise RuntimeError('File block token readback mismatch')
    if uploaded.get('name') != receipt['file_name']:
        raise RuntimeError('File name readback mismatch')
    # A file may be nested in a view block (type 33). Check its owning root too.
    anchor_id = receipt['file_block_id']
    if receipt['file_block_id'] not in after[len(before):]:
        parent = file_block.get('parent_id')
        if parent not in after[len(before):]:
            raise RuntimeError('Appended file root missing')
        if receipt['file_block_id'] not in get_block(client, doc, parent).get('children', []):
            raise RuntimeError('File is not attached to its view root')
        anchor_id = parent
    size, digest = download_digest(client, receipt['file_token'])
    if size != receipt['file_size_bytes'] or digest != receipt['file_sha256']:
        raise RuntimeError('Downloaded attachment differs from local video')
    receipt['status'] = 'verified'
    receipt['video_anchor_url'] = f"{receipt['url']}#{anchor_id}"
    receipt['verification'] = {
        'original_root_prefix_preserved': True,
        'original_root_count': len(before),
        'text_blocks_checked': len(receipt['text_blocks']),
        'file_block_token_and_name_checked': True,
        'downloaded_size_bytes': size,
        'downloaded_sha256': digest,
        'readback_time_utc': datetime.now(timezone.utc).isoformat(),
    }


def text_block(block_type: int, field: str, text: str) -> dict:
    return {'block_type': block_type, field: {'elements': [{'text_run': {'content': text}}]}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path)
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--document-id', default='I2GAdun3nofskUxAOjWcxVwjnJO')
    p.add_argument('--title', default='2026-10-23 三型号固定腰仿真视频')
    p.add_argument('--caption', default='160、132、100 三型号各自摆位，腰部全程固定；'
                   '视频回放已检查的非接触往返轨迹，终点为旧名义配合位置前 30 毫米，'
                   '不是实机运动或插接成功录像。')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if not args.verify_only:
        if args.video is None:
            p.error('--video is required unless --verify-only')
        if args.receipt.exists():
            p.error('Receipt exists; use --verify-only to avoid duplicate append')
        if args.video.suffix.lower() != '.mp4' or not args.video.is_file():
            p.error('--video must name an existing MP4 file')
        if not 0 < args.video.stat().st_size <= 20 * 1024 * 1024:
            p.error('Video must be nonempty and at most 20 MiB for upload_all')
    credentials_path = Path(os.environ.get(
        'FEISHU_CREDENTIALS_FILE', '/home/wjr/.config/feishu-docs/credentials.json'))
    credentials = json.loads(credentials_path.read_text())
    if credentials.get('editor_open_id'):
        os.environ.setdefault('FEISHU_ALLOWED_OPEN_IDS', credentials['editor_open_id'])
    client = FeishuClient(Config(require_webhook=False))
    if args.verify_only:
        receipt = json.loads(args.receipt.read_text())
    else:
        video = args.video.resolve()
        receipt = {'status': 'prepared', 'document_id': args.document_id,
                   'url': f'https://feishu.cn/docx/{args.document_id}',
                   'video_path': str(video), 'file_name': video.name,
                   'file_size_bytes': video.stat().st_size,
                   'file_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
                   'original_root_ids': root_ids(client, args.document_id),
                   'text_blocks': []}
        save(args.receipt, receipt)
        try:
            receipt['status'] = 'appending_text'
            save(args.receipt, receipt)
            specs = [(4, 'heading2', args.title), (2, 'text', args.caption)]
            data = client._request(
                'POST', f'/docx/v1/documents/{args.document_id}/blocks/{args.document_id}/children',
                client._json_body({'children': [text_block(*spec) for spec in specs], 'index': -1}),
                query={'document_revision_id': '-1'})
            receipt['text_blocks'] = [{'block_id': block['block_id'], 'field': spec[1], 'text': spec[2]}
                                      for block, spec in zip(data['children'], specs, strict=True)]
            receipt['status'] = 'creating_file_block'
            save(args.receipt, receipt)
            receipt['file_block_id'] = client.create_file_block(args.document_id)
            receipt['status'] = 'uploading_video'
            save(args.receipt, receipt)
            receipt['file_token'] = client.upload_file(video, receipt['file_block_id'])
            receipt['status'] = 'binding_file_token'
            save(args.receipt, receipt)
            client.replace_file(args.document_id, receipt['file_block_id'], receipt['file_token'])
            receipt['status'] = 'attached'
            save(args.receipt, receipt)
        except Exception as exc:
            receipt['error'] = str(exc)
            save(args.receipt, receipt)
            raise
    verify(client, receipt)
    save(args.receipt, receipt)
    print(json.dumps(receipt['verification'], ensure_ascii=False))
    print(receipt['video_anchor_url'])


if __name__ == '__main__':
    main()
