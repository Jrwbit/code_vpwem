# iPhone Duo 256GB 购买助手

## 推荐：北京门店库存轮询与到店取货

`apple_pickup_bot.py` 是独立运行的版本。它调用 Apple 官网库存接口，列出北京全部 Apple Store 和各 SKU 状态，持续轮询；发现库存后自动加入购物袋、选择取货流程并推进到付款/最终确认页面。它不会点击可能直接扣款的最终按钮。

### 图形控制台

启动本地前端：

```bash
cd /home/wjr/code/vpwem/tools/iphone_duo_reserver
.venv/bin/python pickup_dashboard.py
```

浏览器会打开 `http://127.0.0.1:8765`。控制台显示北京各门店实时库存、运行阶段和日志，并提供登录、开始、停止以及切换到付款窗口的按钮。账号登录和付款只在弹出的 Apple 官方 Chrome 窗口进行，控制台不会接触密码或付款信息。

先用 iPhone 18 Pro Max 256GB 勃艮第酒红色测试：

```bash
cd /home/wjr/code/vpwem/tools/iphone_duo_reserver
.venv/bin/python apple_pickup_bot.py --preset iphone18-test --init
.venv/bin/python apple_pickup_bot.py --preset iphone18-test
```

只查询一次并列出门店，不加入购物袋：

```bash
.venv/bin/python apple_pickup_bot.py --preset iphone18-test --dry-run --once
```

iPhone Duo 正式开售时改用：

```bash
.venv/bin/python apple_pickup_bot.py --preset duo
```

内置目标 SKU：

- iPhone 18 Pro Max 256GB 勃艮第酒红色：`MJY84CH/A`
- iPhone Duo 256GB 星光白色：`MK2M4CH/A`
- iPhone Duo 256GB 夜空色：`MK2N4CH/A`

脚本使用 Apple 官网的 `/shop/retail/pickup-message` 接口，并以北京邮编 `100000` 查询，再过滤出北京市内门店。如果接口异常，脚本会明确记录并继续重试，不会把它当成无货；如果加入购物袋后出现错误页，也不会误报成功。

面向 Apple 中国官网，颜色不限。脚本会在开售时间后循环检查页面，选择 `256GB` 和任一可用颜色，尝试加入购物袋并推进到结账阶段。

脚本也会自动选择“不折抵换购”和“不加 AppleCare+”，并在点击后核验购物袋数量，避免把官网错误页误报为成功。

它不会绕过验证码、风控或排队，也不会点击最终的“下单”或付款按钮。遇到验证时会响铃并等待人工处理。Apple 可能在开售前调整页面，因此请在预购前试运行，并保持浏览器可见。

## 安装

建议使用独立虚拟环境：

```bash
cd /home/wjr/code/vpwem/tools/iphone_duo_reserver
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

## 首次准备

提前运行一次初始化，在打开的浏览器中登录 Apple 账户，并确认账户中的联系电话和收货信息：

```bash
.venv/bin/python iphone_duo_reserver.py --init
```

完成后回到终端按回车。登录状态只保存在本机 `.browser-profile/` 中，不要把该目录分享或提交到 Git。

## 开售日运行

默认时间已设为北京时间 `2026-10-16 20:00`：

```bash
.venv/bin/python iphone_duo_reserver.py
```

建议至少提前 10 分钟启动。脚本会提前打开商品页并等待，到点后每 3–5 秒刷新一次。成功加入购物袋后会截图并响铃，然后尝试打开结账页，最终确认和付款由你手动完成。

自定义开售时间（必须包含时区）：

```bash
.venv/bin/python iphone_duo_reserver.py --launch-at '2026-10-16T20:00:00+08:00'
```

运行记录中的截图保存在 `screenshots/`。如网站临时修改页面导致无法识别，可把失败截图发来调整选择器。
