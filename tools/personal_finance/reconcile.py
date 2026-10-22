"""Evidence-preserving personal cash-flow reconciliation (all money in cents).

The raw export direction is not an economic classification. In particular,
Alipay refunds are neutral, and a closed purchase can have a settled refund.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import re


APP_SOURCES = {"wechat", "alipay", "douyin"}
KINDS = {"income", "expense", "refund", "reimbursement", "internal", "excluded", "review"}


def refund_record(r):
    return (r.get("status") == "退款成功" or
            (r["direction"] != "out" and "退款" in r.get("transaction_type", "")) or
            (r["direction"] == "in" and any(s in r.get("description", "") for s in ("退款", "退货", "退费"))))


def original_id(r):
    return re.split(r"[_*]", r.get("transaction_id", ""))[0]


def find_refunds(records):
    """Link Alipay's explicit parent ID; do not guess by amount alone."""
    orders = defaultdict(list)
    for r in records:
        if not refund_record(r) and r.get("transaction_id"):
            orders[(r["source"], original_id(r))].append(r)
    links = {}
    for r in records:
        if refund_record(r) and r["source"] == "alipay":
            candidates = orders.get((r["source"], original_id(r)), [])
            events = {(p.get("transaction_id"), p["time"], p["direction"], p["amount"]) for p in candidates}
            parent = candidates[0] if len(events) == 1 else None
            if parent and parent["id"] != r["id"]:
                links[r["id"]] = parent["id"]
    return links


def category(r):
    explicit = r.get("transaction_type", "")
    if explicit in {"餐饮美食", "日用百货", "交通出行", "生活服务", "文化休闲", "医疗健康",
                    "商业服务", "教育培训", "酒店旅游", "数码电器", "运动户外", "家居家装",
                    "充值缴费", "美容美发", "服饰装扮", "爱车养车", "公共服务"}:
        return explicit
    text = r.get("description", "") + r.get("counterparty", "")
    for label, pattern in (("医疗健康", "医院|药房|挂号"), ("交通出行", "铁路|火车|地铁|打车|滴滴|高铁|公交|骑行|航空"),
                           ("餐饮美食", "餐|饭|烧烤|火锅|麦当劳|肯德基|奶茶|咖啡|烧肉|猪手|食堂"),
                           ("教育培训", "校园卡|学费"), ("文化休闲", "滑雪|博物馆|电影|门票|旅游")):
        if re.search(pattern, text):
            return label
    return "其他"


def classify(r, refunded_orders, owner_names=(), group_as_reimbursement=True):
    text = " ".join(str(r.get(k, "")) for k in ("transaction_type", "description", "counterparty"))
    payment = r.get("payment_method", "")
    direction = r["direction"]
    status = r.get("status", "")
    result = dict(kind="review", category=category(r), reason="原始记录不足以判断经济性质", review=True,
                  review_reason="请核对款项性质", confidence="low")

    def set_kind(kind, reason, label=None, review=False, review_reason=""):
        result.update(kind=kind, reason=reason, review=review, review_reason=review_reason,
                      confidence="medium" if review else "high")
        if label:
            result["category"] = label
        return result

    if refund_record(r):
        return set_kind("refund", "已实际退款，按退款到账月冲减支出", "消费退款")
    if status in {"交易关闭", "支付失败", "已取消", "交易失败"} and r["id"] not in refunded_orders:
        return set_kind("excluded", "交易未完成，且无关联退款证明曾付款", "未完成交易")
    if "亲情卡" in payment or "他人代付" in payment:
        if re.search("信用购.*还款|花呗.*还款", text):
            return set_kind("internal", "他人代付还款；需核对原消费，避免重复计入", "代付还款", True, "原消费是否包含在本次账单")
        return set_kind("expense", "按用户口径：为本人购买的商品包含亲情卡/他人代付", category(r))
    if re.search("购汇|结汇", text):
        return set_kind("internal", "不同币种之间兑换，不是消费或新增收入", "外汇兑换")
    if "找人周转" in payment and direction == "neutral":
        return set_kind("expense", "周转资金支付商品，商品消费计入；资金来源另列待核", category(r), True,
                        "需确认周转资金是借款还是代付，商品支出保留")
    if "收益发放" in text or (direction == "in" and re.search("结息|利息入账|存款利息", text)):
        return set_kind("income", "已到账收益/利息，不将投资本金计为收入", "利息收益")
    if any(word in text for word in ("零钱充值", "零钱提现", "余额宝-", "余额宝提现", "转账收款到余额宝", "银行卡定时转入")):
        return set_kind("internal", "本人钱包/存款/余额宝之间转移，不计消费或收入", "账户内转")
    if r.get("transaction_type") == "投资理财" or re.search("基金.*买入|基金.*卖出|赎回|理财申购", text):
        return set_kind("internal", "投资本金流转不计消费或收入；卖出损益需成本资料另核", "投资流转")
    if re.search("信用卡.*还款|信用购.*还款|花呗.*还款|借呗.*还款|月付.*还款", text):
        return set_kind("internal", "偿还本金不重复记消费，费用和利息需另核", "还款", True, "需核对是否含利息或手续费及原消费是否齐全")
    if r.get("counterparty", "").strip() in owner_names:
        return set_kind("internal", "交易对方与本次账户持有人全名一致", "本人转账")
    if re.search("借款|贷款|周转|借钱|还钱", text):
        return set_kind("review", "疑似借贷往来，无法仅凭流水确认本金与收益", "借贷待核", True, "需确认本金、利息或其他用途")
    if r.get("transaction_type") == "群收款" and direction == "in" and group_as_reimbursement:
        return set_kind("reimbursement", "暂按 AA 回款抵减本人支出；需确认对应垫付", "AA回款", True, "群收款是否为 AA，以及对应垫付是否包含在账单中")
    if direction == "in":
        transfer = r.get("transaction_type") in {"转账", "转账红包", "二维码收款"} or bool(re.search("汇款|转账|转入", text))
        return set_kind("income", "外部已到账款项；退款、本人互转及投资本金另分", "转账收入" if transfer else "其他收入",
                        transfer, "需确认是否为 AA、借款或本人其他账户转入" if transfer else "")
    if direction == "out":
        transfer = r.get("transaction_type") in {"转账", "转账红包"} or bool(re.search("汇款|转账|转出", text))
        return set_kind("expense", "已付款；关联退款另在到账月抵减" if r["id"] in refunded_orders else "已发生的对外付款",
                        "转账支出" if transfer else None, transfer,
                        "需确认是否为借贷、本人的其他账户或为他人垫付" if transfer else "")
    if "哈啰骑行卡" in payment:
        return set_kind("excluded", "骑行卡权益扣次，无本次现金支出", "权益消费")
    return result


def _timestamp(r):
    return datetime.fromisoformat(r["time"])


def _channel_match(bank, app):
    text = " ".join(str(bank.get(k, "")) for k in ("description", "counterparty", "transaction_type")) + str(bank.get("raw", {}))
    words = {"wechat": ("微信", "财付通", "腾讯"), "alipay": ("支付宝", "蚂蚁"), "douyin": ("抖音", "抖店", "合众")}
    if any(x in text for x in words.get(app["source"], ())):
        return True
    known_channel = any(x in text for values in words.values() for x in values) or any(x in text for x in ("付费通", "钱袋宝", "美团支付", "拼多多支付", "京东支付", "网银在线"))
    return (not known_channel and bool(re.search("快捷|银联|无卡", text))
            and _merchant_compatible(bank, app))


def _account_match(bank, app):
    tail = str(bank.get("account_tail", ""))
    method = app.get("payment_method", "")
    if app["source"] == "douyin" and method == "退款":
        return True  # refund export omits destination; uniqueness across banks is still required
    name = bank.get("bank_name", "")
    if name and "银行" in method and name not in method:
        return False
    if tail:
        return tail in method
    return bool(name and name in method)


def _merchant_compatible(bank, app):
    bank_party = bank.get("counterparty", "").strip()
    app_party = app.get("counterparty", "").strip()
    # Payment processors provide no conflicting merchant identity.
    generic = ("支付宝", "财付通", "微信", "腾讯", "合众", "抖音支付", "银联", "UNIONPAY", "群收款")
    if not bank_party or any(word in bank_party for word in generic):
        return True
    if "*" in app_party or app.get("transaction_type") in {"扫二维码付款", "群收款", "转账"}:
        return True  # masked shop names / nicknames are not conflicting legal identities
    if not app_party or bank_party in app_party or app_party in bank_party:
        return True
    aliases = (("拼多多", "寻梦"), ("美团", "三快"), ("铁路", "12306"), ("盒马",),
               ("所见所得", "抖音"), ("链动",), ("基金销售", "余额宝"))
    if any(any(x in bank_party for x in group) and any(x in app_party for x in group) for group in aliases):
        return True
    # Platforms may display their underlying merchant only in the other
    # export's description; exact merchant text there still corroborates it.
    return bank_party in app.get("description", "") or app_party in bank.get("description", "")


def _economic_direction(r):
    if r["kind"] in {"refund", "income", "reimbursement"}:
        return "in"
    if r["kind"] == "expense":
        return "out"
    if r["kind"] == "internal":
        if re.search("转出到银行卡|提现", r.get("description", "") + r.get("transaction_type", "")):
            return "in"  # bank leg of a wallet withdrawal
        return "out"
    return r["direction"]


def _time_compatible(bank, app):
    delta = (_timestamp(bank) - _timestamp(app)).total_seconds()
    if bank.get("time_precision") == "day":
        return bank["time"][:10] == app["time"][:10]
    if bank["source"] == "bank_boc" and 0 <= delta <= 6 * 3600:
        # This export repeatedly posts overnight batches after the purchase;
        # account/channel/amount and mutual uniqueness remain required.
        return True
    return bank["time"][:10] == app["time"][:10] and abs(delta) <= 300


def reconcile(records, owner_names=(), group_as_reimbursement=True):
    records = deepcopy(records)
    refund_links = find_refunds(records)
    refunded_orders = set(refund_links.values())
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Raw record IDs must be unique")
    matches = []
    transactions = []
    removed = set()
    order_seen = {}
    for r in records:
        r.update(classify(r, refunded_orders, owner_names, group_as_reimbursement))
        r["funded_by_other"] = any(x in r.get("payment_method", "") for x in ("亲情卡", "他人代付"))
        r["month"] = r["time"][:7]
        r["sources"] = [r["source"]]
        r["evidence"] = [deepcopy(by_id[r["id"]])]
        # Same export row repeated in another overlapping export: same order,
        # timestamp, direction and amount. Refund event IDs remain distinct.
        key = (r["source"], r.get("transaction_id"), r["time"], r["direction"], r["amount"])
        if r.get("transaction_id") and key in order_seen:
            parent = order_seen[key]
            parent["evidence"].extend(r["evidence"])
            removed.add(r["id"])
            matches.append(dict(id=f"repeat:{r['id']}", type="duplicate", amount=r["amount"],
                                reason="同来源交易号、时间、方向及金额完全一致", record_ids=[parent["id"], r["id"]]))
        else:
            order_seen[key] = r
            transactions.append(r)
    for refund_id, parent_id in refund_links.items():
        by_id[refund_id]["related_id"] = parent_id
        matches.append(dict(id=f"refund:{refund_id}", type="refund_link", amount=by_id[refund_id]["amount"],
                            reason="支付宝退款号包含原交易号；收付分别按发生月计入", record_ids=[parent_id, refund_id]))

    from .douyin_audit import annotate_douyin_chains
    chains = annotate_douyin_chains(records)
    for chain in chains["chains"]:
        if not chain["balanced"]:
            continue
        parent = by_id[chain["original_id"]]
        if chain["onp_outgoing_ids"]:
            parent.update(review=True, confidence="medium", review_reason="月付核销去重依据整组金额闭环推断，非完整订单外键")
        for child_id in chain["onp_outgoing_ids"]:
            child = by_id[child_id]
            parent["evidence"].extend(child["evidence"])
            removed.add(child_id)
            matches.append(dict(id=f"douyin-confirm:{child_id}", type="duplicate", amount=child["amount"],
                                reason="同商户月付原单与后续核销/退款整组金额闭环；核销不重复计消费（推断）",
                                record_ids=[parent["id"], child_id], chain=chain))
    transactions = [r for r in transactions if r["id"] not in removed]

    # Reconcile bank postings to the purchase record only when the payment
    # account and channel agree as well as direction, date and exact cents.
    # A many-to-many collision is exposed for review, never greedily collapsed.
    apps = [r for r in transactions if r["source"] in APP_SOURCES and r["kind"] not in {"excluded", "review"}]
    banks = [r for r in transactions if r["source"] not in APP_SOURCES]
    candidates = {}
    reverse = defaultdict(list)
    for bank in banks:
        choices = []
        for app in apps:
            if (bank["amount"] != app["amount"] or bank["direction"] != _economic_direction(app)
                    or not _time_compatible(bank, app)
                    or not _account_match(bank, app) or not _channel_match(bank, app)
                    or not _merchant_compatible(bank, app)):
                continue
            choices.append(app)
        # Prefer an exact posted second when nearby identical amounts exist.
        # This is still a mutual one-to-one assignment, never a greedy pairing.
        exact = [a for a in choices if abs((_timestamp(bank) - _timestamp(a)).total_seconds()) <= 1]
        if bank.get("time_precision") != "day" and len(exact) == 1:
            choices = exact
        candidates[bank["id"]] = choices
        for app in choices:
            reverse[app["id"]].append(bank)
    for bank in banks:
        choices = candidates[bank["id"]]
        if len(choices) == 1 and len(reverse[choices[0]["id"]]) == 1:
            app = choices[0]
            app["evidence"].extend(bank["evidence"])
            app["sources"].append(bank["source"])
            removed.add(bank["id"])
            delayed = abs((_timestamp(bank) - _timestamp(app)).total_seconds()) > 300 and bank.get("time_precision") != "day"
            unknown_account = app["source"] == "douyin" and app.get("payment_method") == "退款"
            reason = "渠道、方向、金额及付款账户一致，时间相符且唯一对应"
            if delayed:
                reason = "中行延迟记账6小时内；同渠道、方向、金额且双向唯一"
                app["confidence"] = "medium"
            if unknown_account:
                reason += "；抖音退款不含卡号，由唯一银行入账交叉佐证"
                app["confidence"] = "medium"
            matches.append(dict(id=f"bank:{bank['id']}", type="duplicate", amount=bank["amount"],
                                reason=reason, record_ids=[app["id"], bank["id"]]))
        elif choices:
            bank.update(kind="review", review=True, confidence="low", category="重复待核",
                        reason="同金额候选不唯一；暂不重复计入，保留待核", review_reason="需确认对应 App 交易")
            bank["candidate_ids"] = [r["id"] for r in choices]

    # A balanced same-month block can be deduplicated as a block even if two
    # identical refunds cannot be assigned individually. Keep that limitation.
    for bank in banks:
        choices = candidates[bank["id"]]
        if bank["id"] in removed or len(choices) < 2:
            continue
        bank_group = reverse[choices[0]["id"]]
        app_ids = {a["id"] for a in choices}
        bank_ids = {b["id"] for b in bank_group}
        if (len(bank_group) != len(choices) or any(b["id"] in removed for b in bank_group)
                or any({a["id"] for a in candidates[b["id"]]} != app_ids for b in bank_group)
                or any({b["id"] for b in reverse[a["id"]]} != bank_ids for a in choices)
                or len({r["month"] for r in bank_group + choices}) != 1
                or len({b["source"] for b in bank_group}) != 1):
            continue
        for a in choices:
            a.update(confidence="medium", review=True, review_reason="同月同额退款组总额与笔数闭环，组内逐笔顺序不能唯一确认")
            a["sources"] = sorted(set(a["sources"] + [bank["source"]]))
        for b in bank_group:
            choices[0]["evidence"].extend(b["evidence"])
            removed.add(b["id"])
        matches.append(dict(id=f"bank-group:{bank['id']}", type="duplicate_group",
                            amount=sum(b["amount"] for b in bank_group),
                            reason="同月同渠道同额记录组，笔数/总额闭环；按组去重，未强行逐笔配对",
                            record_ids=sorted(app_ids | bank_ids)))

    # Split tender: the app records the full purchase while the bank records
    # only its share. This rule applies to explicit wallet + card payments.
    mixed = [a for a in apps if a["source"] == "douyin" and "抖音零钱&" in a.get("payment_method", "")]
    split_parents = {}
    split_candidates = {}
    split_reverse = defaultdict(list)
    for bank in banks:
        if bank["id"] in removed or bank["direction"] != "out":
            continue
        possible = [a for a in mixed if 0 < bank["amount"] < a["amount"]
                    and _account_match(bank, a) and _channel_match(bank, a) and _time_compatible(bank, a)]
        split_candidates[bank["id"]] = possible
        for a in possible:
            split_reverse[a["id"]].append(bank)
    for bank in banks:
        possible = split_candidates.get(bank["id"], [])
        if len(possible) != 1 or len(split_reverse[possible[0]["id"]]) != 1:
            continue
        parent = possible[0]
        split_parents[parent["id"]] = bank
        parent["evidence"].extend(bank["evidence"])
        parent["sources"].append(bank["source"])
        removed.add(bank["id"])
        matches.append(dict(id=f"split:{bank['id']}", type="duplicate", amount=bank["amount"],
                            reason="抖音明确标记零钱+银行卡组合支付；银行只记部分扣款，整单支出保留",
                            record_ids=[parent["id"], bank["id"]]))
    refund_parts = {}
    refund_reverse = defaultdict(list)
    for bank in banks:
        if bank["id"] in removed or bank["direction"] != "in":
            continue
        possible = []
        for refund in apps:
            if refund["source"] != "douyin" or refund["kind"] != "refund" or not _channel_match(bank, refund) or not _time_compatible(bank, refund):
                continue
            parents = [a for a in mixed if a["id"] in split_parents and a["amount"] == refund["amount"]
                       and a["transaction_id"][-4:] == refund["transaction_id"][-4:]
                       and a["counterparty"] == refund["counterparty"] and a["time"] < refund["time"]
                       and split_parents[a["id"]]["amount"] == bank["amount"]]
            if len(parents) == 1:
                possible.append(refund)
        refund_parts[bank["id"]] = possible
        for refund in possible:
            refund_reverse[refund["id"]].append(bank)
    for bank in banks:
        possible = refund_parts.get(bank["id"], [])
        if len(possible) == 1 and len(refund_reverse[possible[0]["id"]]) == 1:
            refund = possible[0]
            refund["evidence"].extend(bank["evidence"])
            refund["sources"].append(bank["source"])
            refund["confidence"] = "medium"
            removed.add(bank["id"])
            matches.append(dict(id=f"split-refund:{bank['id']}", type="duplicate", amount=bank["amount"],
                                reason="组合支付全额退回：原单/退款总额、尾码及时间一致，银行退款恰等于原银行扣款部分",
                                record_ids=[refund["id"], bank["id"]]))

    # If only a smaller bank refund and a larger app refund remain, retain the
    # unproven bank leg visibly for review instead of subtracting it twice.
    for bank in banks:
        if bank["id"] in removed or bank["direction"] != "in" or bank["kind"] != "refund":
            continue
        possible = [a for a in apps if a["source"] == "douyin" and a["kind"] == "refund"
                    and bank["amount"] < a["amount"] and _channel_match(bank, a) and _time_compatible(bank, a)]
        if len(possible) == 1:
            bank.update(kind="review", category="部分退款待核", review=True, confidence="low",
                        reason="同时间抖音整单退款金额较大，疑似银行子金额；缺原单，暂不重复抵减",
                        review_reason="需确认是否属于抖音退款的一部分", candidate_ids=[possible[0]["id"]])

    for bank in banks:
        if bank["id"] in removed or bank["direction"] != "in" or bank["kind"] != "income":
            continue
        name = re.sub(r"[（）()]", "", bank.get("counterparty", ""))
        if "旅行社" not in name:
            continue
        prior = [b for b in banks if b["direction"] == "out" and b["amount"] >= bank["amount"]
                 and re.sub(r"[（）()]", "", b.get("counterparty", "")) == name
                 and 0 <= (_timestamp(bank) - _timestamp(b)).total_seconds() <= 7 * 86400]
        if len(prior) == 1:
            bank.update(kind="refund", category="旅行退款", review=True, confidence="medium",
                        reason="同旅行社收款后7天内返还部分金额，暂按退款冲减",
                        review_reason="请确认旅行社返款为退款", related_id=prior[0]["id"])

    transactions = [r for r in transactions if r["id"] not in removed]
    transactions.sort(key=lambda r: (r["time"], r["id"]), reverse=True)
    return transactions, matches


def monthly_totals(transactions):
    months = defaultdict(lambda: {k: 0 for k in ("income", "expense", "refund", "reimbursement", "internal", "excluded",
                                                 "review_in", "review_out", "review_neutral", "review_count")})
    for r in transactions:
        month = months[r["time"][:7]]
        if r["kind"] == "review":
            month[f"review_{r['direction']}"] += r["amount"]
        else:
            month[r["kind"]] += r["amount"]
        month["review_count"] += bool(r["review"])
    for m in months.values():
        m["net_expense"] = m["expense"] - m["refund"] - m["reimbursement"]
        m["balance"] = m["income"] - m["net_expense"]
    return dict(sorted(months.items()))


def fingerprint(records):
    text = "\n".join(sorted(f"{r['id']}|{r['amount']}|{r['time']}" for r in records))
    return hashlib.sha256(text.encode()).hexdigest()[:20]
