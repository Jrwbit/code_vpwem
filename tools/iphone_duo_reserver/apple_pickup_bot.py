#!/usr/bin/env python3
"""Apple 中国官网：北京门店库存轮询与到店取货下单辅助。

独立运行，不依赖对话控制。检测到门店库存后会自动加入购物袋并推进到
付款/最终确认页面；不会点击可能直接扣款的最终按钮，也不会绕过验证码。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import BrowserContext, Locator, Page, TimeoutError, sync_playwright


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / ".pickup-browser-profile"
SCREENSHOT_DIR = ROOT / "screenshots"
INVENTORY_ENDPOINT = "https://www.apple.com.cn/shop/retail/pickup-message"
PAYMENT_WORDS = re.compile(r"支付|付款|Apple\s*Pay|提交订单|下单", re.I)
ERROR_WORDS = re.compile(r"Page Not Found|页面未找到|找不到.*页面|抱歉，出错了", re.I)


@dataclass(frozen=True)
class Part:
    sku: str
    label: str
    url: str


@dataclass(frozen=True)
class Preset:
    name: str
    parts: tuple[Part, ...]


PRESETS = {
    "iphone18-test": Preset(
        name="iPhone 18 Pro Max 256GB 勃艮第酒红色",
        parts=(
            Part(
                sku="MJY84CH/A",
                label="勃艮第酒红色",
                url="https://www.apple.com.cn/shop/buy-iphone/iphone-18-pro/mjy84ch/a",
            ),
        ),
    ),
    "duo": Preset(
        name="iPhone Duo 256GB（颜色不限）",
        parts=(
            Part(
                sku="MK2M4CH/A",
                label="星光白色",
                url="https://www.apple.com.cn/shop/buy-iphone/iphone-duo/mk2m4ch/a",
            ),
            Part(
                sku="MK2N4CH/A",
                label="夜空色",
                url="https://www.apple.com.cn/shop/buy-iphone/iphone-duo/mk2n4ch/a",
            ),
        ),
    ),
}


class InventoryError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}", flush=True)


def notify(message: str) -> None:
    print("\a", end="", flush=True)
    try:
        subprocess.run(
            ["notify-send", "Apple 到店取货助手", message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


def screenshot(page: Page, label: str) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.png"
    try:
        page.screenshot(path=str(path), full_page=True)
    except Exception as exc:
        log(f"截图失败：{exc}")
    return path


def open_context(playwright, profile: Path) -> BrowserContext:
    profile.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        user_data_dir=str(profile),
        headless=False,
        viewport={"width": 1440, "height": 960},
        locale="zh-CN",
        timezone_id="Asia/Shanghai",
    )
    if shutil.which("google-chrome") or shutil.which("google-chrome-stable"):
        kwargs["channel"] = "chrome"
    return playwright.chromium.launch_persistent_context(**kwargs)


def inventory_url(parts: tuple[Part, ...], location: str) -> str:
    params: list[tuple[str, str]] = [("location", location)]
    params.extend((f"parts.{index}", part.sku) for index, part in enumerate(parts))
    return f"{INVENTORY_ENDPOINT}?{urlencode(params)}"


def fetch_inventory(page: Page, parts: tuple[Part, ...], location: str) -> dict:
    url = inventory_url(parts, location)
    result = page.evaluate(
        """async (url) => {
            const response = await fetch(url, {
                credentials: 'include',
                headers: {'Accept': 'application/json'}
            });
            return {status: response.status, text: await response.text()};
        }""",
        url,
    )
    if result["status"] != 200:
        raise InventoryError(f"库存接口 HTTP {result['status']}")
    try:
        return json.loads(result["text"])
    except json.JSONDecodeError as exc:
        raise InventoryError("库存接口没有返回 JSON") from exc


def lookup_availability(store: dict, sku: str) -> dict:
    availability = store.get("partsAvailability", {})
    candidates = (sku, sku.upper(), sku.replace("/", ""), sku.split("/")[0])
    for key in candidates:
        if key in availability:
            return availability[key]
    for key, value in availability.items():
        if key.replace("/", "").upper() == sku.replace("/", "").upper():
            return value
    return {}


def store_name(store: dict) -> str:
    return str(store.get("storeName") or store.get("storeDisplayName") or store.get("storeNumber") or "未知门店")


def inventory_rows(payload: dict, parts: tuple[Part, ...]) -> list[dict]:
    body = payload.get("body", {}) if isinstance(payload, dict) else {}
    stores = body.get("stores")
    if stores is None:
        stores = body.get("content", {}).get("pickupMessage", {}).get("stores", [])
    rows = []
    for store in stores:
        # 邮编查询会返回京外的邻近门店；本任务只监控北京市内门店。
        if store.get("state") != "北京" and store.get("city") != "北京":
            continue
        for part in parts:
            availability = lookup_availability(store, part.sku)
            regular = availability.get("messageTypes", {}).get("regular", {})
            enabled = bool(
                availability.get("storeSelectionEnabled")
                or regular.get("storeSelectionEnabled")
            )
            rows.append(
                {
                    "store": store,
                    "store_name": store_name(store),
                    "store_number": str(store.get("storeNumber") or ""),
                    "part": part,
                    "available": enabled,
                    "quote": str(
                        availability.get("pickupSearchQuote")
                        or availability.get("storePickupQuote")
                        or regular.get("storePickupQuote")
                        or availability.get("pickupDisplay")
                        or ("可取货" if enabled else "不可取货")
                    ),
                }
            )
    return rows


def print_inventory(rows: list[dict]) -> None:
    if not rows:
        log("库存接口未返回门店列表")
        return
    print("\n北京 Apple Store 到店取货状态", flush=True)
    print("-" * 78, flush=True)
    for row in rows:
        mark = "有货" if row["available"] else "无货"
        print(
            f"{row['store_name']:<24} {row['store_number']:<8} "
            f"{row['part'].label:<10} {mark:<4} {row['quote']}",
            flush=True,
        )
    print("-" * 78, flush=True)


def row_signature(rows: list[dict]) -> tuple:
    return tuple(
        (row["store_number"], row["part"].sku, row["available"], row["quote"])
        for row in rows
    )


def click_input_label(page: Page, selector: str, name: str) -> bool:
    item = page.locator(selector).first
    try:
        if item.count() == 0 or not item.is_enabled():
            return False
        input_id = item.get_attribute("id")
        if not input_id:
            return False
        page.locator(f'label[for="{input_id}"]').click(force=True, timeout=2500)
        page.wait_for_timeout(500)
        log(f"已选择{name}")
        return item.is_checked()
    except Exception:
        return False


def select_no_extras(page: Page) -> None:
    if not click_input_label(page, "#noTradeIn", "不折抵换购"):
        click_input_label(page, '[data-autom="choose-noTradeIn"]', "不折抵换购")
    click_input_label(page, '[data-autom="noapplecare"]', "不加 AppleCare+")


def combined_text(item: Locator) -> str:
    try:
        return " ".join(
            f'{item.inner_text(timeout=300) or ""} {item.get_attribute("aria-label") or ""}'.split()
        )
    except Exception:
        return ""


def bag_count(page: Page) -> int | None:
    items = page.locator('button, [role="button"], a')
    for index in range(items.count()):
        text = combined_text(items.nth(index))
        match = re.search(r"(\d+)\s*\+?\s*打开购物袋", text)
        if match:
            return int(match.group(1))
    return None


def clickable(page: Page, pattern: re.Pattern[str]) -> list[Locator]:
    items = page.locator('button, [role="button"], [role="radio"], label, a')
    found: list[tuple[int, Locator]] = []
    for index in range(items.count()):
        item = items.nth(index)
        try:
            text = combined_text(item)
            if pattern.search(text) and item.is_visible() and item.is_enabled():
                found.append((len(text), item))
        except Exception:
            continue
    return [item for _, item in sorted(found, key=lambda pair: pair[0])]


def click_text(page: Page, pattern: re.Pattern[str], name: str) -> bool:
    for item in clickable(page, pattern):
        text = combined_text(item)
        if PAYMENT_WORDS.search(text):
            continue
        try:
            item.scroll_into_view_if_needed(timeout=1000)
            item.click(timeout=2500)
            page.wait_for_timeout(900)
            log(f"已点击{name}：{text[:80]}")
            return True
        except Exception:
            continue
    return False


def add_to_bag(page: Page, part: Part) -> bool:
    page.goto(part.url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(1200)
    select_no_extras(page)
    page.wait_for_timeout(700)
    before = bag_count(page) or 0
    if not click_text(page, re.compile(r"添加到购物袋|加入购物袋"), "加入购物袋"):
        log("加入购物袋按钮尚不可用")
        return False
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except TimeoutError:
        pass
    page.wait_for_timeout(1500)
    try:
        if ERROR_WORDS.search(f"{page.title()}\n{page.locator('body').inner_text(timeout=1500)}"):
            log("官网返回错误页，商品没有成功加入购物袋")
            return False
    except Exception:
        return False
    after = bag_count(page)
    return bool((after is not None and after > before) or "/shop/bag" in page.url)


def fill_pickup_search(page: Page, location: str) -> bool:
    inputs = page.locator("input")
    for index in range(inputs.count()):
        item = inputs.nth(index)
        try:
            descriptor = " ".join(
                f'{item.get_attribute("placeholder") or ""} '
                f'{item.get_attribute("aria-label") or ""} '
                f'{item.get_attribute("name") or ""}'.split()
            )
            if item.is_visible() and re.search(r"地点|位置|城市|邮政编码|搜索|location|postal", descriptor, re.I):
                item.fill(location)
                item.press("Enter")
                page.wait_for_timeout(1500)
                log(f"已搜索取货地点：{location}")
                return True
        except Exception:
            continue
    return False


def at_payment_boundary(page: Page) -> bool:
    try:
        body = page.locator("body").inner_text(timeout=1500)
    except Exception:
        return False
    return bool(re.search(r"付款方式|支付方式|核对订单|确认订单", body))


def advance_to_payment(page: Page, row: dict, location: str) -> bool:
    """推进取货结账流程；遇到最终付款边界立即停止。"""
    if "/shop/bag" not in page.url:
        click_text(page, re.compile(r"打开购物袋|查看购物袋"), "打开购物袋")

    for _ in range(12):
        if at_payment_boundary(page):
            return True
        if ERROR_WORDS.search(page.title()):
            return False

        progressed = False
        progressed |= click_text(page, re.compile(r"到店取货|店内取货|我要取货|选择取货"), "到店取货")
        if fill_pickup_search(page, location):
            progressed = True
        progressed |= click_text(page, re.compile(re.escape(row["store_name"])), "目标门店")
        progressed |= click_text(page, re.compile(r"选择此.*店|在此店取货|确认取货"), "确认门店")
        progressed |= click_text(page, re.compile(r"继续结账|结账"), "结账")
        progressed |= click_text(page, re.compile(r"^继续$|继续以访客身份"), "继续")

        if not progressed:
            break
    return at_payment_boundary(page)


def initialize(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    with sync_playwright() as playwright:
        context = open_context(playwright, args.profile)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(preset.parts[0].url, wait_until="domcontentloaded", timeout=60_000)
        log("请在浏览器中登录 Apple 账户，并确认姓名、电话等资料。完成后回终端按回车。")
        input()
        context.close()
    return 0


def run(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    log(f"目标：{preset.name}；地点：{args.location}")
    with sync_playwright() as playwright:
        context = open_context(playwright, args.profile)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(preset.parts[0].url, wait_until="domcontentloaded", timeout=60_000)

        previous_signature = None
        while True:
            try:
                payload = fetch_inventory(page, preset.parts, args.location)
                rows = inventory_rows(payload, preset.parts)
                signature = row_signature(rows)
                if signature != previous_signature:
                    print_inventory(rows)
                    previous_signature = signature

                available = [row for row in rows if row["available"]]
                if available:
                    chosen = available[0]
                    log(
                        f"发现库存：{chosen['store_name']} / "
                        f"{chosen['part'].label} / {chosen['part'].sku}"
                    )
                    notify(f"{chosen['store_name']} 有货，正在进入取货结账流程")
                    if args.dry_run:
                        context.close()
                        return 0
                    if add_to_bag(page, chosen["part"]):
                        reached = advance_to_payment(page, chosen, args.location)
                        path = screenshot(page, "pickup-checkout")
                        if reached:
                            notify("已到付款/最终确认页面，请立即接管")
                            log(f"已停止在付款边界；截图：{path}")
                        else:
                            notify("已发现库存，但自动结账需要人工接管")
                            log(f"结账页面结构未能继续识别；截图：{path}")
                        input("浏览器将保持打开，处理完成后按回车退出：")
                        context.close()
                        return 0
                    log("库存出现但加入购物袋失败，将继续轮询")

                if args.once:
                    context.close()
                    return 1 if not available else 0
            except InventoryError as exc:
                log(f"{exc}；稍后重试（这不代表门店无货）")
                if args.once:
                    context.close()
                    return 2
            except Exception as exc:
                log(f"本轮出错：{type(exc).__name__}: {exc}")
                if args.once:
                    context.close()
                    return 3

            delay = random.uniform(args.min_interval, args.max_interval)
            time.sleep(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apple 北京门店库存轮询与到店取货助手")
    parser.add_argument("--preset", choices=PRESETS, default="iphone18-test")
    parser.add_argument("--location", default="100000", help="北京查询邮编")
    parser.add_argument("--profile", type=Path, default=PROFILE_DIR)
    parser.add_argument("--min-interval", type=float, default=5.0)
    parser.add_argument("--max-interval", type=float, default=8.0)
    parser.add_argument("--init", action="store_true", help="登录并保存浏览器会话")
    parser.add_argument("--dry-run", action="store_true", help="只列门店和库存，不加入购物袋")
    parser.add_argument("--once", action="store_true", help="只查询一次，用于测试")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.min_interval < 3:
        parser.error("最短轮询间隔不能低于 3 秒")
    if args.max_interval < args.min_interval:
        parser.error("max-interval 必须大于或等于 min-interval")
    return initialize(args) if args.init else run(args)


if __name__ == "__main__":
    sys.exit(main())
