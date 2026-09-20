#!/usr/bin/env python3
"""Apple 到店取货本地控制台。访问 http://127.0.0.1:8765。"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

import apple_pickup_bot as bot


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apple 到店取货控制台</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#121722;--line:#273044;--muted:#93a0b5;--blue:#56a8ff;--green:#39d98a;--red:#ff6678;--amber:#ffc857}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#17233a 0,#090b10 38%);color:#edf3ff;font:15px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}
main{max-width:1180px;margin:auto;padding:28px 20px 60px}h1{font-size:28px;margin:0 0 6px}.sub{color:var(--muted);margin-bottom:24px}.toolbar,.summary,.logs{background:rgba(18,23,34,.92);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:18px}
.toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:grid;gap:6px;color:var(--muted)}select,input{background:#0b0f17;border:1px solid #354057;color:#fff;border-radius:9px;padding:10px 12px;font:inherit}
button{border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer;background:#29344a;color:#fff}button.primary{background:#1677ff}button.danger{background:#a92739}button:disabled{opacity:.45;cursor:not-allowed}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:99px;padding:6px 11px}.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}.running .dot{background:var(--green);box-shadow:0 0 12px var(--green)}.error .dot{background:var(--red)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px;margin:16px 0 20px}.card{background:rgba(18,23,34,.92);border:1px solid var(--line);border-radius:14px;padding:16px}.card.available{border-color:#238c60;box-shadow:0 0 0 1px #238c6044}.store{font-size:18px;font-weight:750}.meta{color:var(--muted);font-size:13px}.stock{font-size:18px;font-weight:800;margin-top:12px}.yes{color:var(--green)}.no{color:var(--red)}
.kv{display:grid;grid-template-columns:130px 1fr;gap:8px 15px}.kv div:nth-child(odd){color:var(--muted)}pre{white-space:pre-wrap;margin:0;max-height:280px;overflow:auto;color:#c8d3e6;font:13px/1.55 ui-monospace,monospace}.notice{color:var(--amber)}
@media(max-width:620px){.kv{grid-template-columns:1fr}.kv div:nth-child(odd){margin-top:7px}.toolbar>*{width:100%}}
</style></head><body><main>
<h1>Apple 到店取货控制台</h1><div class="sub">实时库存、自动加入购物袋与付款接管。付款始终在 Apple 官方 Chrome 窗口完成。</div>
<section class="toolbar">
 <label>目标<select id="preset"><option value="iphone18-test">iPhone 18 Pro Max 红色 256GB（测试）</option><option value="duo">iPhone Duo 256GB（颜色不限）</option></select></label>
 <label>北京查询邮编<input id="location" value="100000" inputmode="numeric"></label>
 <label style="display:flex;align-items:center;gap:8px;padding:10px"><input id="dry" type="checkbox" style="width:auto">仅监控，不下单</label>
 <button id="login">① 初始化/登录</button><button id="loginDone">登录完成</button>
 <button id="start" class="primary">② 开始监控并抢购</button><button id="focus">打开付款窗口</button><button id="stop" class="danger">停止</button>
</section>
<section class="summary"><div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap"><span id="pill" class="pill"><span class="dot"></span><span id="mode">空闲</span></span><span id="checked" class="meta">尚未查询</span></div>
<div class="kv" style="margin-top:16px"><div>目标</div><div id="target">—</div><div>阶段</div><div id="stage">—</div><div>最近错误</div><div id="error">—</div><div>付款说明</div><div class="notice">检测到库存后会弹出真实 Apple Chrome。控制台不代理账号或付款信息。</div></div></section>
<h2>北京门店库存</h2><div id="stores" class="grid"><div class="card meta">等待首次查询…</div></div>
<section class="logs"><h2 style="margin-top:0">运行日志</h2><pre id="logs">等待启动…</pre></section>
</main><script>
const $=id=>document.getElementById(id); async function post(path,data={}){return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(r=>r.json())}
$('login').onclick=()=>post('/api/login',{preset:$('preset').value}); $('loginDone').onclick=()=>post('/api/login-done');
$('start').onclick=()=>post('/api/start',{preset:$('preset').value,location:$('location').value,dry_run:$('dry').checked});
$('stop').onclick=()=>post('/api/stop'); $('focus').onclick=()=>post('/api/focus');
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function refresh(){try{let s=await fetch('/api/status').then(r=>r.json());$('mode').textContent=s.mode_label;$('target').textContent=s.target||'—';$('stage').textContent=s.stage||'—';$('error').textContent=s.error||'—';$('checked').textContent=s.last_check?'最近查询：'+s.last_check:'尚未查询';
let active=['monitoring','checkout','payment','login'].includes(s.mode);$('pill').className='pill '+(active?'running':s.mode==='error'?'error':'');$('start').disabled=active;$('login').disabled=active;
$('stores').innerHTML=s.stores.length?s.stores.map(x=>`<div class="card ${x.available?'available':''}"><div class="store">${esc(x.store_name)}</div><div class="meta">${esc(x.store_number)} · ${esc(x.address)}</div><div class="meta">${esc(x.part_label)} · ${esc(x.sku)}</div><div class="stock ${x.available?'yes':'no'}">${x.available?'● 可取货':'○ 暂不可取'}</div><div>${esc(x.quote)}</div></div>`).join(''):'<div class="card meta">等待库存数据…</div>';
$('logs').textContent=s.logs.join('\n')||'等待启动…';$('logs').scrollTop=$('logs').scrollHeight}catch(e){}finally{setTimeout(refresh,1000)}}refresh();
</script></body></html>"""


class Dashboard:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "mode": "idle",
            "mode_label": "空闲",
            "target": "",
            "stage": "等待启动",
            "error": "",
            "last_check": "",
            "stores": [],
            "logs": [],
            "payment_url": "",
        }
        self.logs: deque[str] = deque(maxlen=250)
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.login_done_event = threading.Event()
        self.focus_event = threading.Event()

    def log(self, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        with self.lock:
            self.logs.append(line)
            self.state["logs"] = list(self.logs)
        print(line, flush=True)

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.state)
            result["stores"] = list(self.state["stores"])
            result["logs"] = list(self.logs)
            return result

    def busy(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def launch(self, target, *args) -> tuple[bool, str]:
        if self.busy():
            return False, "已有任务正在运行"
        self.stop_event.clear()
        self.login_done_event.clear()
        self.focus_event.clear()
        self.worker = threading.Thread(target=target, args=args, daemon=True)
        self.worker.start()
        return True, "已启动"

    def login(self, preset_name: str) -> tuple[bool, str]:
        if preset_name not in bot.PRESETS:
            return False, "未知目标"
        return self.launch(self._login_worker, preset_name)

    def _login_worker(self, preset_name: str) -> None:
        preset = bot.PRESETS[preset_name]
        self.update(mode="login", mode_label="等待登录", target=preset.name, stage="请在弹出的 Apple Chrome 中登录")
        self.log("已打开专用 Apple Chrome；登录后在控制台点击“登录完成”")
        try:
            with sync_playwright() as playwright:
                context = bot.open_context(playwright, bot.PROFILE_DIR)
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(preset.parts[0].url, wait_until="domcontentloaded", timeout=60_000)
                while not self.stop_event.is_set() and not self.login_done_event.wait(0.5):
                    if self.focus_event.is_set():
                        page.bring_to_front()
                        self.focus_event.clear()
                context.close()
            if self.login_done_event.is_set():
                self.log("登录会话已保存")
                self.update(mode="idle", mode_label="就绪", stage="可以开始监控")
            else:
                self.update(mode="stopped", mode_label="已停止", stage="登录初始化已取消")
        except Exception as exc:
            self.log(f"登录窗口异常：{exc}")
            self.update(mode="error", mode_label="错误", error=str(exc), stage="登录初始化失败")

    @staticmethod
    def serialize_rows(rows: list[dict]) -> list[dict]:
        result = []
        for row in rows:
            address = row["store"].get("address", {})
            result.append(
                {
                    "store_name": row["store_name"],
                    "store_number": row["store_number"],
                    "part_label": row["part"].label,
                    "sku": row["part"].sku,
                    "available": row["available"],
                    "quote": row["quote"],
                    "address": address.get("address2") or address.get("postalCode") or "北京",
                }
            )
        return result

    def start(self, preset_name: str, location: str, dry_run: bool) -> tuple[bool, str]:
        if preset_name not in bot.PRESETS:
            return False, "未知目标"
        if not location.isdigit() or len(location) != 6:
            return False, "请输入 6 位北京邮编"
        return self.launch(self._monitor_worker, preset_name, location, dry_run)

    def _monitor_worker(self, preset_name: str, location: str, dry_run: bool) -> None:
        preset = bot.PRESETS[preset_name]
        self.update(
            mode="monitoring", mode_label="监控中", target=preset.name,
            stage="轮询北京门店", error="", stores=[], payment_url="",
        )
        self.log(f"开始监控：{preset.name}，邮编 {location}，间隔 5–8 秒")
        previous = None
        try:
            with sync_playwright() as playwright:
                context = bot.open_context(playwright, bot.PROFILE_DIR)
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(preset.parts[0].url, wait_until="domcontentloaded", timeout=60_000)
                while not self.stop_event.is_set():
                    if self.focus_event.is_set():
                        page.bring_to_front()
                        self.focus_event.clear()
                    try:
                        payload = bot.fetch_inventory(page, preset.parts, location)
                        rows = bot.inventory_rows(payload, preset.parts)
                        signature = bot.row_signature(rows)
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.update(last_check=now, stores=self.serialize_rows(rows), error="")
                        if signature != previous:
                            available_count = sum(row["available"] for row in rows)
                            self.log(f"库存更新：北京 {len({r['store_number'] for r in rows})} 家门店，{available_count} 个可取货组合")
                            previous = signature
                        available = [row for row in rows if row["available"]]
                        if available:
                            chosen = available[0]
                            self.log(f"发现库存：{chosen['store_name']} / {chosen['part'].label}")
                            bot.notify(f"{chosen['store_name']} 有货")
                            if dry_run:
                                self.update(mode="idle", mode_label="发现库存", stage="仅监控模式，未下单")
                                break
                            self.update(mode="checkout", mode_label="自动结账", stage=f"正在锁定 {chosen['store_name']}")
                            if bot.add_to_bag(page, chosen["part"]):
                                reached = bot.advance_to_payment(page, chosen, location)
                                shot = bot.screenshot(page, "dashboard-checkout")
                                stage = "已到付款/最终确认页，请在 Apple Chrome 中付款" if reached else "需要在 Apple Chrome 中人工接管"
                                self.log(f"{stage}；截图 {shot.name}")
                                self.update(mode="payment", mode_label="等待付款", stage=stage, payment_url=page.url)
                                bot.notify("请接管 Apple Chrome 完成付款")
                                page.bring_to_front()
                                while not self.stop_event.wait(0.5):
                                    if self.focus_event.is_set():
                                        page.bring_to_front()
                                        self.focus_event.clear()
                                break
                            self.log("发现库存，但加入购物袋失败；恢复轮询")
                            self.update(mode="monitoring", mode_label="监控中", stage="库存瞬时消失或官网拒绝，继续轮询")
                    except bot.InventoryError as exc:
                        self.update(error=str(exc), last_check=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        self.log(f"库存接口异常：{exc}")
                    except Exception as exc:
                        self.update(error=f"{type(exc).__name__}: {exc}")
                        self.log(f"本轮异常：{type(exc).__name__}: {exc}")
                    self.stop_event.wait(5 + (time.time() % 3))
                context.close()
            if self.stop_event.is_set():
                self.update(mode="stopped", mode_label="已停止", stage="任务已停止")
        except Exception as exc:
            self.log(f"监控任务失败：{type(exc).__name__}: {exc}")
            self.update(mode="error", mode_label="错误", error=str(exc), stage="监控进程退出")


DASHBOARD = Dashboard()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        pass

    def send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, status: int, data: dict) -> None:
        self.send_bytes(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_bytes(200, HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self.json_response(200, DASHBOARD.snapshot())
        else:
            self.json_response(404, {"ok": False, "message": "not found"})

    def read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return {}

    def do_POST(self) -> None:
        data = self.read_json()
        if self.path == "/api/start":
            ok, message = DASHBOARD.start(
                str(data.get("preset", "iphone18-test")),
                str(data.get("location", "100000")),
                bool(data.get("dry_run", False)),
            )
        elif self.path == "/api/login":
            ok, message = DASHBOARD.login(str(data.get("preset", "iphone18-test")))
        elif self.path == "/api/login-done":
            DASHBOARD.login_done_event.set()
            ok, message = True, "正在保存登录会话"
        elif self.path == "/api/stop":
            DASHBOARD.stop_event.set()
            ok, message = True, "正在停止"
        elif self.path == "/api/focus":
            DASHBOARD.focus_event.set()
            ok, message = True, "正在切换到 Apple Chrome"
        else:
            self.json_response(404, {"ok": False, "message": "not found"})
            return
        self.json_response(200 if ok else 409, {"ok": ok, "message": message})


def main() -> int:
    parser = argparse.ArgumentParser(description="Apple 到店取货本地控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"控制台已启动：{url}", flush=True)
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        DASHBOARD.stop_event.set()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
