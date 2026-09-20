#!/usr/bin/env python3
"""Apple 中国官网 iPhone Duo 256GB 购买辅助脚本。

脚本不会绕过验证码、风控或排队，也不会点击最终的下单/付款按钮。
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import BrowserContext, Locator, Page, TimeoutError, sync_playwright


DEFAULT_URL = "https://www.apple.com.cn/shop/buy-iphone/iphone-duo"
DEFAULT_LAUNCH_AT = "2026-10-16T20:00:00+08:00"
PROFILE_DIR = Path(__file__).resolve().parent / ".browser-profile"
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"

CAPACITY_RE = re.compile(r"256\s*GB", re.I)
COLOR_PATTERNS = (re.compile(r"夜空色"), re.compile(r"星光白(?:色)?"))
ADD_PATTERNS = (
    re.compile(r"添加到购物袋|加入购物袋"),
    re.compile(r"预购"),
)
CHECKOUT_PATTERNS = (
    re.compile(r"查看购物袋"),
    re.compile(r"(?:继续)?结账"),
)
BLOCK_PATTERNS = re.compile(r"验证码|安全验证|机器人|排队|请验证|captcha", re.I)
FAILURE_PATTERNS = re.compile(r"Page Not Found|页面未找到|找不到.*页面|抱歉，出错了", re.I)


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] {message}", flush=True)


def notify(message: str) -> None:
    """尽最大努力发出桌面通知和终端响铃。"""
    print("\a", end="", flush=True)
    try:
        subprocess.run(
            ["notify-send", "iPhone Duo 购买助手", message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


def save_screenshot(page: Page, label: str) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.png"
    try:
        page.screenshot(path=str(path), full_page=True)
    except Exception as exc:  # 截图失败不应终止抢购流程
        log(f"截图失败：{exc}")
    return path


def parse_launch_at(value: str) -> datetime:
    launch_at = datetime.fromisoformat(value)
    if launch_at.tzinfo is None:
        raise argparse.ArgumentTypeError("开售时间必须带时区，例如 +08:00")
    return launch_at


def wait_until(launch_at: datetime) -> None:
    while True:
        remaining = (launch_at - datetime.now().astimezone()).total_seconds()
        if remaining <= 0:
            return
        if remaining > 120:
            sleep_for = min(30.0, remaining - 90)
        elif remaining > 10:
            sleep_for = min(2.0, remaining - 8)
        else:
            sleep_for = min(0.2, remaining)
        log(f"距离开售还有 {remaining:.1f} 秒") if remaining > 10 else None
        time.sleep(max(0.05, sleep_for))


def interactive_candidates(page: Page, pattern: re.Pattern[str]) -> list[Locator]:
    selectors = "button, [role=button], [role=radio], label, a"
    locator = page.locator(selectors).filter(has_text=pattern)
    result: list[tuple[int, Locator]] = []
    for index in range(locator.count()):
        item = locator.nth(index)
        try:
            if not item.is_visible() or not item.is_enabled():
                continue
            text = (item.inner_text(timeout=500) or item.get_attribute("aria-label") or "").strip()
            result.append((len(text), item))
        except Exception:
            continue
    return [item for _, item in sorted(result, key=lambda pair: pair[0])]


def click_first(page: Page, patterns: tuple[re.Pattern[str], ...], name: str) -> bool:
    for pattern in patterns:
        for item in interactive_candidates(page, pattern):
            try:
                item.scroll_into_view_if_needed(timeout=1000)
                item.click(timeout=2000)
                log(f"已点击{name}：{pattern.pattern}")
                page.wait_for_timeout(700)
                return True
            except Exception:
                continue
    return False


def select_capacity(page: Page) -> bool:
    return click_first(page, (CAPACITY_RE,), "256GB 容量")


def select_any_color(page: Page) -> str | None:
    # 两个颜色都可接受；每轮轮换优先顺序，避免一直卡在某个无货颜色。
    patterns = list(COLOR_PATTERNS)
    random.shuffle(patterns)
    for pattern in patterns:
        if click_first(page, (pattern,), "颜色"):
            return "夜空色" if "夜空" in pattern.pattern else "星光白色"
    return None


def click_input_label(page: Page, selector: str, name: str) -> bool:
    """点击被美化组件遮住的 radio input 所对应的 label。"""
    item = page.locator(selector).first
    try:
        if item.count() == 0 or not item.is_enabled():
            return False
        input_id = item.get_attribute("id")
        if not input_id:
            return False
        page.locator(f'label[for="{input_id}"]').click(force=True, timeout=2000)
        page.wait_for_timeout(500)
        log(f"已选择{name}")
        return item.is_checked()
    except Exception:
        return False


def select_no_extras(page: Page) -> None:
    """Apple 购买页要求显式选择这两项后才会启用购物袋按钮。"""
    if not click_input_label(page, '#noTradeIn', "不折抵换购"):
        click_input_label(page, '[data-autom="choose-noTradeIn"]', "不折抵换购")
    click_input_label(page, '[data-autom="noapplecare"]', "不加 AppleCare+")


def bag_count(page: Page) -> int | None:
    locator = page.locator('button, [role="button"], a')
    for index in range(locator.count()):
        try:
            item = locator.nth(index)
            text = " ".join(
                f'{item.inner_text(timeout=300) or ""} {item.get_attribute("aria-label") or ""}'.split()
            )
            if "打开购物袋" not in text:
                continue
            match = re.search(r"(\d+)\s*\+?\s*打开购物袋", text)
            if match:
                return int(match.group(1))
        except Exception:
            continue
    return None


def add_to_bag_and_verify(page: Page) -> bool:
    """点击加入购物袋，并用购物袋数量或购物袋页面确认结果。"""
    before = bag_count(page)
    if not click_first(page, ADD_PATTERNS, "加入购物袋/预购"):
        return False
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except TimeoutError:
        pass
    page.wait_for_timeout(1200)

    try:
        title = page.title()
        body = page.locator("body").inner_text(timeout=2000)
        if FAILURE_PATTERNS.search(f"{title}\n{body}"):
            log("加入购物袋请求被官网错误页拒绝，未判定为成功")
            return False
    except Exception:
        return False

    after = bag_count(page)
    if after is not None and after > (before or 0):
        return True
    if "/shop/bag" in page.url and not FAILURE_PATTERNS.search(title):
        return True
    log(f"未检测到购物袋数量增加（操作前={before}，操作后={after}）")
    return False


def page_is_blocked(page: Page) -> bool:
    try:
        body = page.locator("body").inner_text(timeout=2000)
    except Exception:
        return False
    return bool(BLOCK_PATTERNS.search(body))


def attempt_reservation(page: Page) -> tuple[bool, str | None]:
    """尝试选中任意颜色的 256GB，并加入购物袋。"""
    # 官网组件的出现顺序可能变化，所以容量/颜色交替尝试两轮。
    selected_color = None
    for _ in range(2):
        select_capacity(page)
        selected_color = select_any_color(page) or selected_color
        select_capacity(page)

    if not selected_color:
        return False, None

    select_no_extras(page)
    if add_to_bag_and_verify(page):
        return True, selected_color
    return False, selected_color


def advance_to_checkout(page: Page) -> None:
    """最多推进至结账页面，绝不点击最终下单或付款。"""
    for _ in range(3):
        if not click_first(page, CHECKOUT_PATTERNS, "结账入口"):
            break
        page.wait_for_timeout(1200)
    log("已停止自动操作：请在浏览器中核对商品、地址并手动完成验证和付款。")


def open_context(playwright, profile: Path) -> BrowserContext:
    profile.mkdir(parents=True, exist_ok=True)
    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile),
        headless=False,
        viewport={"width": 1440, "height": 960},
        locale="zh-CN",
        timezone_id="Asia/Shanghai",
        args=["--disable-blink-features=AutomationControlled"],
    )


def init_profile(url: str, profile: Path) -> int:
    with sync_playwright() as playwright:
        context = open_context(playwright, profile)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        log("浏览器已打开。请登录 Apple 账户、确认默认收货资料，然后回到终端按回车保存会话。")
        input()
        context.close()
    return 0


def run(args: argparse.Namespace) -> int:
    with sync_playwright() as playwright:
        context = open_context(playwright, args.profile)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        log(f"商品页已打开：{args.url}")
        wait_until(args.launch_at)

        attempt = 0
        while True:
            attempt += 1
            try:
                if attempt > 1:
                    # 上轮加入购物袋可能被官网导向错误页，始终回到规范商品页重试。
                    page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
                log(f"第 {attempt} 次检查库存")

                if page_is_blocked(page):
                    path = save_screenshot(page, "verification")
                    notify("页面要求验证，请立即手动接管浏览器")
                    log(f"检测到验证码、排队或安全验证；截图：{path}")
                    input("完成验证后按回车继续监控：")
                    continue

                added, color = attempt_reservation(page)
                if added:
                    path = save_screenshot(page, "added-to-bag")
                    notify(f"已将 iPhone Duo 256GB（{color}）加入购物袋，请立即确认")
                    log(f"成功加入购物袋：256GB {color}；截图：{path}")
                    advance_to_checkout(page)
                    notify("已到结账阶段，请手动确认并付款")
                    input("浏览器将保持打开；操作完成后按回车退出：")
                    context.close()
                    return 0
            except TimeoutError as exc:
                log(f"页面超时，将重试：{exc}")
            except Exception as exc:
                log(f"本轮失败，将重试：{type(exc).__name__}: {exc}")

            delay = random.uniform(args.min_interval, args.max_interval)
            log(f"暂未成功，{delay:.1f} 秒后重试")
            time.sleep(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apple 中国官网 iPhone Duo 256GB 购买辅助脚本")
    parser.add_argument("--init", action="store_true", help="首次使用：打开浏览器并保存登录会话")
    parser.add_argument("--url", default=DEFAULT_URL, help="Apple 中国官网商品页")
    parser.add_argument("--launch-at", type=parse_launch_at, default=parse_launch_at(DEFAULT_LAUNCH_AT))
    parser.add_argument("--profile", type=Path, default=PROFILE_DIR, help="浏览器会话目录")
    parser.add_argument("--min-interval", type=float, default=3.0, help="最短刷新间隔（秒）")
    parser.add_argument("--max-interval", type=float, default=5.0, help="最长刷新间隔（秒）")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.min_interval < 2:
        parser.error("为避免给官网造成过高请求压力，最短刷新间隔不能低于 2 秒")
    if args.max_interval < args.min_interval:
        parser.error("max-interval 必须大于或等于 min-interval")
    if args.init:
        return init_profile(args.url, args.profile)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
