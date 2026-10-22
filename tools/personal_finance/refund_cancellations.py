"""Exclude evidenced fully refunded purchase groups without deleting records."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import re


def _original_kind(row: dict) -> str:
    return row.get("removal_original_kind", row.get("kind", "")) if row.get("full_refund_removed") else row.get("kind", "")


def _currency(row: dict) -> str:
    return row.get("currency") or "CNY"


def _meaningful_description(row: dict) -> str | None:
    text = re.sub(r"^(?:退款|退货款)[\s:：\-—]*", "", str(row.get("description", "")).strip())
    text = re.sub(r"\s+", "", text)
    party = re.sub(r"\s+", "", str(row.get("counterparty", "")))
    if len(text) < 6 or text == party or re.fullmatch(r"[\d\W]+", text):
        return None
    if re.match(r"(?:网上)?快捷|银联|小额普通|转账备注|收款方备注", text):
        return None
    if text in {"抖音生活服务退款", "抖音电商退款", "平台退款", "商品退款", "平台订单", "退款成功"}:
        return None
    return text


def _bank_merchant_key(row: dict) -> tuple | None:
    """Keep the complete exported name as the key; reject payment platforms."""
    party = str(row.get("counterparty") or "").strip()
    account = str(row.get("account_tail") or "").strip()
    if (not row.get("source", "").startswith("bank_") or not account or account in {"/", "-"}
            or not party or party in {"/", "-"} or row["amount"] <= 0
            or row.get("time_precision") in {"day", "date"}):
        return None
    # Prefixes are inspected only to judge specificity, never removed from the
    # matching key. Two different bank descriptions must remain different keys.
    merchant = party.rsplit("-", 1)[-1].strip()
    if (re.search(r"平台商户|平台交易|支付机构|京东商城业务|美团汉海信息技术|抖音生活服务|抖音电商|财付通支付科技", merchant)
            or re.fullmatch(r"(?:微信|支付宝|财付通|银联|网银在线|美团|钱袋宝|抖音|拼多多|京东|付费通|拉卡拉)(?:支付|商城)?", merchant)
            or re.match(r"^(?:支付宝|网银在线|银联商务|北京钱袋宝|上海付费通|拉卡拉|连连支付|京东支付|抖音支付).*(?:公司|科技|服务|技术)$", merchant)
            or not re.search(r"店|商行|超市|商场|酒店|宾馆|餐厅|餐饮|饭|烧肉|火锅|早餐|百货|药房|药业|医院|门诊|公司|中心|学校", merchant)):
        return None
    summary = str(row.get("raw", {}).get("summary") or row.get("transaction_type") or row.get("description") or "")
    text = " ".join(str(row.get(key) or "") for key in ("counterparty", "description", "category", "transaction_type"))
    if re.search(r"转账|汇款|汇入|借款|还款|还贷|贷款|借出|借入|红包|充值|提现|内部划转", text):
        return None
    kind = _original_kind(row)
    if kind == "expense" and (row.get("direction") != "out" or not re.search(r"快捷支付|网上支付|商户支付|消费|购物|POS", summary, re.I)):
        return None
    if kind == "refund" and (row.get("direction") != "in" or "退款" not in summary):
        return None
    return row["source"], account, _currency(row), party, row["amount"]


def remove_fully_refunded(transactions: list[dict], matches: list[dict] | None = None,
                          douyin_chains: dict | list[dict] | None = None) -> dict:
    """Mark a whole purchase/refund group excluded only at exact zero net.

    Explicit parent links take precedence. Supplemental links use complete order
    IDs, specific item descriptions, WeChat full-refund statuses with a complete
    group, revalidated Douyin monthly-pay audit chains, or a same-account bank
    purchase/refund with a specific merchant and a unique 24-hour candidate.
    Equal amounts alone
    never establish a relationship; ``douyin_chains`` accepts the audit dict or
    its ``chains`` list.
    """
    by_id = {row["id"]: row for row in transactions}
    if len(by_id) != len(transactions):
        raise ValueError("Canonical transaction IDs must be unique")
    aliases = defaultdict(set)
    for row in transactions:
        aliases[row["id"]].add(row["id"])
        for evidence in row.get("evidence", []):
            if evidence.get("id"):
                aliases[evidence["id"]].add(row["id"])

    def resolve(identifier: str) -> set[str]:
        return {identifier} if identifier in by_id else aliases.get(identifier, set())

    expenses = [row for row in transactions if _original_kind(row) == "expense"]
    refunds = [row for row in transactions if _original_kind(row) == "refund"]
    match_parents = defaultdict(set)
    for match in matches or []:
        if match.get("type") != "refund_link":
            continue
        resolved = set().union(*(resolve(identifier) for identifier in match.get("record_ids", [])))
        parents = {identifier for identifier in resolved if _original_kind(by_id[identifier]) == "expense"}
        for identifier in resolved:
            if _original_kind(by_id[identifier]) == "refund":
                match_parents[identifier].update(parents)

    assignments = defaultdict(dict)
    blocked = set()
    unresolved, conflicts, invalid = set(), set(), set()
    unlinked = []
    for refund in refunds:
        related = refund.get("related_id")
        proposals = set(match_parents.get(refund["id"], set()))
        if related:
            targets = resolve(related)
            proposals.update(targets)
            if len(targets) != 1:
                blocked.update(proposals)
                unresolved.add(refund["id"])
                continue
        if not proposals:
            if related:
                unresolved.add(refund["id"])
            else:
                unlinked.append(refund)
            continue
        if len(proposals) != 1:
            conflicts.add(refund["id"])
            blocked.update(proposals)
            continue
        parent = by_id[next(iter(proposals))]
        if (_original_kind(parent) != "expense" or _currency(parent) != _currency(refund)
                or not parent.get("time") or not refund.get("time") or parent["time"] > refund["time"]):
            invalid.add(refund["id"])
            blocked.add(parent["id"])
            continue
        assignments[parent["id"]][refund["id"]] = "明确原单关联"

    for refund in unlinked:
        merchant_id = refund.get("merchant_id")
        description = _meaningful_description(refund)
        candidates = []
        for parent in expenses:
            if (parent.get("source") != refund.get("source") or _currency(parent) != _currency(refund)
                    or not parent.get("time") or not refund.get("time") or parent["time"] > refund["time"]
                    or refund["amount"] > parent["amount"]):
                continue
            if (merchant_id and merchant_id not in {"/", "-"} and len(merchant_id) >= 6
                    and parent.get("merchant_id") == merchant_id):
                candidates.append((parent, "同来源完整商户订单号一致"))
            elif (description and description == _meaningful_description(parent)
                  and parent.get("counterparty") and parent.get("counterparty") == refund.get("counterparty")):
                candidates.append((parent, "同来源同商户且具体商品描述完全一致"))
        if len(candidates) == 1:
            parent, method = candidates[0]
            assignments[parent["id"]][refund["id"]] = method
        elif len(candidates) > 1:
            conflicts.add(refund["id"])
            blocked.update(parent["id"] for parent, _ in candidates)
        else:
            unresolved.add(refund["id"])

    groups, consumed = [], set()

    def exclude_group(purchases: list[dict], group_refunds: list[dict], method: str) -> None:
        record_ids = sorted(row["id"] for row in purchases) + sorted(row["id"] for row in group_refunds)
        digest = hashlib.sha256("\n".join(sorted(record_ids)).encode()).hexdigest()[:20]
        group_id = "full-refund:" + digest
        members = purchases + group_refunds
        cross_month = len({row["time"][:7] for row in members}) > 1
        reason = ("用户口径：原支出与关联退款累计金额完全相同，整组从支出和退款统计中排除；"
                  + method
                  + ("；包含跨月退款" if cross_month else ""))
        for row in members:
            if not row.get("full_refund_removed"):
                row["removal_original_kind"] = row["kind"]
                row["removal_original_reason"] = row.get("reason", "")
                row["removal_original_review"] = row.get("review", False)
                row["removal_original_review_reason"] = row.get("review_reason", "")
            row.update(kind="excluded", full_refund_removed=True, full_refund_group=group_id,
                       review=False, review_reason="", reason=reason)
        consumed.update(record_ids)
        groups.append({"id": group_id, "record_ids": record_ids, "expense": sum(row["amount"] for row in purchases),
                       "refund": sum(row["amount"] for row in group_refunds), "currency": _currency(purchases[0]), "reason": reason,
                       "cross_month": cross_month})

    partial_count = over_count = 0
    for parent_id, refund_methods in assignments.items():
        parent = by_id[parent_id]
        group_refunds = [by_id[identifier] for identifier in refund_methods]
        refund_amount = sum(row["amount"] for row in group_refunds)
        if parent_id in blocked:
            continue
        if refund_amount > parent["amount"]:
            over_count += 1
            continue
        if refund_amount != parent["amount"] or parent["amount"] <= 0:
            partial_count += 1
            continue
        exclude_group([parent], group_refunds, "、".join(sorted(set(refund_methods.values()))))

    # WeChat exports no original order ID on refunds. A full-refund status on
    # both sides plus the complete same-party/method/amount group is independent
    # business evidence. Never select a convenient subset or claim a one-to-one
    # order relationship when multiple purchases share the group key.
    status_groups = defaultdict(lambda: {"expense": [], "refund": []})
    for row in expenses + refunds:
        if (row.get("source") != "wechat" or not row.get("counterparty")
                or row["counterparty"] in {"/", "-"} or not row.get("payment_method")
                or row["payment_method"] in {"/", "-"} or row["amount"] <= 0):
            continue
        kind = _original_kind(row)
        if kind == "expense" and (row.get("direction") != "out"
                                    or row.get("status") not in {"已全额退款", "对方已退还"}):
            continue
        key = (_currency(row), row["counterparty"], row["payment_method"], row["amount"])
        status_groups[key][kind].append(row)
    assigned_refunds = {refund_id for links in assignments.values() for refund_id in links}
    for parts in status_groups.values():
        purchases, returns = parts["expense"], parts["refund"]
        ids = {row["id"] for row in purchases + returns}
        if (not purchases or len(purchases) != len(returns)
                or ids & (consumed | blocked | conflicts | invalid)
                or any(row.get("related_id") or row["id"] in assigned_refunds for row in returns)
                or any(row.get("status") != "已全额退款" or row.get("direction") != "in" for row in returns)):
            continue
        originals = sorted(purchases, key=lambda row: row.get("time", ""))
        ordered_returns = sorted(returns, key=lambda row: row.get("time", ""))
        if any(not purchase.get("time") or not refund.get("time") or purchase["time"] > refund["time"]
               for purchase, refund in zip(originals, ordered_returns)):
            continue
        exclude_group(purchases, returns, "微信双方均明确全额退款状态，同商户、支付方式、金额的全部记录数量及时间闭环（组级关联）")
        unresolved.difference_update(row["id"] for row in returns)

    # Recheck already audited monthly-pay chains against current canonical
    # amounts and evidence. Confirmation legs are evidence, never new expenses.
    chain_data = douyin_chains.get("chains", []) if isinstance(douyin_chains, dict) else douyin_chains or []
    ambiguous_originals = {identifier for event in (douyin_chains.get("ambiguous_events", [])
                            if isinstance(douyin_chains, dict) else [])
                          for identifier in event.get("candidate_original_ids", [])}
    chain_refund_users = defaultdict(set)
    for index, chain in enumerate(chain_data):
        for identifier in chain.get("onp_refund_ids", []) + chain.get("inp_refund_ids", []):
            for canonical_id in resolve(identifier):
                chain_refund_users[canonical_id].add(index)
    for chain in chain_data:
        parent_ids = resolve(chain.get("original_id", ""))
        refund_ids = chain.get("onp_refund_ids", []) + chain.get("inp_refund_ids", [])
        if (not chain.get("balanced") or chain.get("original_id") in ambiguous_originals
                or len(parent_ids) != 1 or not refund_ids or len(set(refund_ids)) != len(refund_ids)):
            continue
        parent = by_id[next(iter(parent_ids))]
        if (_original_kind(parent) != "expense" or parent["id"] in consumed | blocked
                or parent.get("source") != "douyin" or parent.get("payment_method") != "抖音月付"
                or parent.get("direction") != "out" or parent.get("merchant_id", "").startswith("ONP")
                or parent["amount"] != chain.get("original_amount")):
            continue
        returns = []
        for identifier in refund_ids:
            candidates = resolve(identifier)
            if len(candidates) != 1:
                break
            refund = by_id[next(iter(candidates))]
            expected_prefix = "ONP" if identifier in chain.get("onp_refund_ids", []) else "INP"
            if (_original_kind(refund) != "refund" or refund["id"] in consumed | conflicts | invalid
                    or len(chain_refund_users[refund["id"]]) != 1
                    or refund["id"] in assigned_refunds or refund.get("related_id")
                    or refund.get("source") != "douyin" or refund.get("direction") != "in"
                    or not refund.get("merchant_id", "").startswith(expected_prefix)):
                break
            returns.append(refund)
        if len(returns) != len(refund_ids) or len({row["id"] for row in returns}) != len(returns):
            continue
        confirmations = []
        for identifier in chain.get("onp_outgoing_ids", []):
            evidence = [row for row in parent.get("evidence", []) if row.get("id") == identifier]
            if (identifier in by_id or resolve(identifier) != {parent["id"]} or len(evidence) != 1
                    or evidence[0].get("source") != "douyin" or evidence[0].get("direction") != "out"
                    or not evidence[0].get("merchant_id", "").startswith("ONP")):
                break
            confirmations.append(evidence[0])
        if len(confirmations) != len(chain.get("onp_outgoing_ids", [])):
            continue
        events = confirmations + returns
        suffix = parent.get("transaction_id", "")[-4:]
        if (len(suffix) != 4 or any(_currency(row) != _currency(parent)
                or row.get("transaction_id", "")[-4:] != suffix
                or row.get("counterparty") != parent.get("counterparty")
                or not parent.get("time") or not row.get("time") or row["time"] < parent["time"]
                for row in events)):
            continue
        onp_refunds = sum(row["amount"] for row in returns if row["merchant_id"].startswith("ONP"))
        if (parent["amount"] <= 0 or sum(row["amount"] for row in returns) != parent["amount"]
                or sum(row["amount"] for row in confirmations) + onp_refunds != parent["amount"]):
            continue
        exclude_group([parent], returns, "抖音月付候选链复核：同商户、单号末4位及时间相符，原付款=核销+撤回且原付款=全部退款；核销仅保留为凭证（非完整订单号关联）")
        unresolved.difference_update(row["id"] for row in returns)

    # Construct the entire candidate graph before accepting a pair. Removing
    # a convenient match first must never turn a 2x2 ambiguity into uniqueness.
    bank_groups = defaultdict(lambda: {"expense": [], "refund": []})
    for row in expenses + refunds:
        key = _bank_merchant_key(row)
        if key is not None:
            bank_groups[key][_original_kind(row)].append(row)
    bank_candidates, bank_forward, bank_reverse = [], defaultdict(list), defaultdict(list)
    for parts in bank_groups.values():
        for purchase in parts["expense"]:
            for refund in parts["refund"]:
                if not purchase.get("time") or not refund.get("time"):
                    continue
                lag = (datetime.fromisoformat(refund["time"]) - datetime.fromisoformat(purchase["time"])).total_seconds()
                if 0 <= lag <= 86400:
                    bank_candidates.append((purchase, refund))
                    bank_forward[purchase["id"]].append(refund["id"])
                    bank_reverse[refund["id"]].append(purchase["id"])
    for purchase, refund in bank_candidates:
        if (len(bank_forward[purchase["id"]]) != 1 or len(bank_reverse[refund["id"]]) != 1
                or {purchase["id"], refund["id"]} & (consumed | blocked | conflicts | invalid)
                or purchase["id"] in assignments or refund["id"] in assigned_refunds or refund.get("related_id")):
            continue
        exclude_group([purchase], [refund], "同银行账户、账单商户字段完全一致，24小时内双向唯一等额退款（业务证据推断，无完整订单号；允许同秒入账）")
        unresolved.discard(refund["id"])
    groups.sort(key=lambda group: group["id"])
    return {
        "groups": groups,
        "summary": {"group_count": len(groups), "record_count": sum(len(group["record_ids"]) for group in groups),
                    "expense_removed": sum(group["expense"] for group in groups),
                    "refund_removed": sum(group["refund"] for group in groups),
                    "cross_month_count": sum(group["cross_month"] for group in groups)},
        "diagnostics": {"unresolved_refund_count": len(unresolved), "conflicting_refund_count": len(conflicts),
                        "invalid_link_count": len(invalid), "partial_group_count": partial_count,
                        "over_refunded_group_count": over_count, "blocked_parent_count": len(blocked)},
        "notes": ["仅排除有可信原单关系且整组退款金额等于支出的记录；不按相同金额猜测原单。",
                  "原始记录、金额、用途分类和凭证均保留，排除前的统计类型与复核依据另存。",
                  "跨月全退整组排除会改变各月毛支出和退款，但所有月份合计净支出不变。"],
    }
