"""Attach searchable counterparties and payment roles to canonical rows.

No amounts or classifications are changed. Evidence supplies names, never extra
transactions; entity totals must be computed from canonical rows exactly once.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
import unicodedata


_SURNAMES = set("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔龚程嵇邢裴陆荣翁荀羊甄曲封储靳段巫焦巴弓牧隗山谷车侯宓蓬全班仰秋仲伊宫宁仇栾暴甘厉戎祖武符刘景詹束龙叶幸司黎印宿白怀蒲邰从鄂索咸赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳姬申扶堵冉宰雍桑桂濮牛寿通边扈燕冀郏浦尚农温庄晏柴瞿阎连茹习艾鱼容向古易廖庾终居衡步都耿满弘匡文寇广禄阙东欧利师巩聂晁勾敖融冷訾辛阚那简饶曾沙养须丰巢关蒯相查后荆红游竺权逯盖益桓公")
_ORGANIZATION = re.compile(r"公司|银行|大学|学校|商户|平台|中心|支付|财付通|支付宝|合作社|旗舰店|便利店|超市|基金|科技|餐厅|饭店|咖啡|门店|客服", re.I)
_TRANSFER_CATEGORIES = {"本人转账", "转账收入", "转账支出", "往来款抵消", "对外转账消费", "借贷待核", "AA回款"}
_EMPTY_METHODS = {"", "-", "--", "/", "退款", "原路退回", "原路退款"}
_COMPOUND_SURNAMES = {"欧阳", "司马", "上官", "诸葛", "夏侯", "东方", "皇甫", "尉迟", "公孙", "慕容", "司徒", "司空", "令狐", "宇文", "长孙", "端木", "南宫"}


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _transfer_context(record: dict) -> bool:
    kind = str(record.get("transaction_type", ""))
    summary = str(record.get("raw", {}).get("summary", ""))
    return (record.get("category") in _TRANSFER_CATEGORIES
            or bool(re.search(r"转账|汇款", kind + " " + summary))
            or kind in {"群收款", "二维码收款"})


def _person_name(record: dict, aliases: dict[str, str]) -> str | None:
    text = _normalize(record.get("counterparty", ""))
    if text in aliases:
        return aliases[text]
    if text and text in aliases.values():
        return text
    if not _transfer_context(record) or _ORGANIZATION.search(text):
        return None
    match = re.match(r"^([\u4e00-\u9fff]{2,4})(?=$|\s|\()", text)
    if match and (len(match.group(1)) <= 3 and match.group(1)[0] in _SURNAMES
                  or match.group(1)[:2] in _COMPOUND_SURNAMES):
        name = match.group(1)
        return aliases.get(name, name)
    if record.get("category") == "本人转账" and re.fullmatch(r"[A-Z]+(?: [A-Z]+){1,3}", text):
        return aliases.get(text, text)
    return None


def annotate_entities(transactions: list[dict], person_aliases: dict[str, str] | None = None) -> list[dict]:
    """Set ``entity_ids`` / ``entity_roles`` in place and return a directory.

    Roles are ``counterparty``, ``payer`` and ``payment``. Explicit aliases are
    matched against complete normalized names, never description substrings.
    Family-card payer names remain separately labelled until explicitly aliased.
    Refunds with a missing payment method inherit it through ``related_id``.
    The six people with the largest canonical participation amounts are featured;
    family cards and explicit payment-on-behalf dimensions are also featured.
    """
    aliases = {_normalize(source): _normalize(target) for source, target in (person_aliases or {}).items()}
    directory: dict[str, dict] = {}
    seen_aliases: dict[str, set] = defaultdict(set)
    index = {}
    for row in transactions:
        index[row["id"]] = row
        for evidence in row.get("evidence", []):
            if evidence.get("id"):
                index[evidence["id"]] = row

    def payment_methods(row: dict, seen: set[str] | None = None) -> list[str]:
        seen = set() if seen is None else seen
        if row["id"] in seen:
            return []
        seen.add(row["id"])
        methods = [_normalize(record.get("payment_method", "")) for record in [row, *row.get("evidence", [])]]
        if (_normalize(row.get("payment_method", "")) in _EMPTY_METHODS
                and row.get("kind") == "refund" and row.get("related_id") in index):
            methods += payment_methods(index[row["related_id"]], seen)
        return methods

    for row in transactions:
        roles: dict[str, set[str]] = defaultdict(set)

        def add(entity_type: str, label: str, role: str, names=(), key: str | None = None,
                featured: bool = False, description: str | None = None) -> str:
            identity = _normalize(key or label)
            identifier = entity_type + ":" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            if identifier not in directory:
                directory[identifier] = {"id": identifier, "label": label, "type": entity_type,
                                         "aliases": [], "featured": featured}
                if description:
                    directory[identifier]["description"] = description
            roles[identifier].add(role)
            seen_aliases[identifier].update(_normalize(name) for name in names if _normalize(name))
            return identifier

        primary = _person_name(row, aliases)
        identified = {}
        for record in [row, *row.get("evidence", [])]:
            name = _person_name(record, aliases)
            if name:
                identified.setdefault(name, set()).add(record.get("counterparty", ""))
        for name, variants in identified.items():
            add("person", name, "counterparty", names=variants)
        party = _normalize(row.get("counterparty", ""))
        if not primary and party and party not in {"/", "-", "--"}:
            if len(identified) == 1 and _transfer_context(row):
                name = next(iter(identified))
                add("person", name, "counterparty", names=[party])
            else:
                add("merchant", party, "counterparty", names=[row.get("counterparty", "")])

        for method in payment_methods(row):
            if "亲情卡" in method:
                add("payment", "亲情卡", "payment", names=["亲戚卡"], featured=True,
                    description="按付款方式识别，包含关联退款；不代表付款人与某个转账人自动同一。")
                match = re.search(r"亲情卡\s*\(\s*([^()]+)", method)
                if match:
                    payer = _normalize(match.group(1))
                    display_match = re.search(r"亲情卡\((.*?)\)(?=&|$)", method)
                    payer_display = display_match.group(1) if display_match else payer
                    if payer in aliases:
                        add("person", aliases[payer], "payer", names=[payer, payer_display, payer + "（亲情卡付款人）"])
                    else:
                        add("person", payer + "（亲情卡付款人）", "payer", names=[payer, payer_display],
                            key="family-card-payer:" + payer,
                            description="来自亲情卡付款方式中的姓名，未与转账对手自动合并。")
            if re.search(r"他人代付|找人代付", method):
                add("payment", "他人代付", "payment", names=["找人代付"], featured=True)
            for label in ("抖音月付", "美团月付", "花呗", "信用购"):
                if label in method:
                    add("payment", label, "payment")

        row["entity_ids"] = sorted(roles)
        row["entity_roles"] = {identifier: sorted(values) for identifier, values in sorted(roles.items())}

    participation = defaultdict(int)
    for row in transactions:
        for identifier in row["entity_ids"]:
            if directory[identifier]["type"] == "person":
                participation[identifier] += row["amount"]
    featured_people = set(sorted(participation, key=lambda identifier: (-participation[identifier], identifier))[:6])
    for identifier, entry in directory.items():
        entry["aliases"] = sorted(seen_aliases[identifier] - {_normalize(entry["label"])})
        if identifier in featured_people:
            entry["featured"] = True
    order = {"person": 0, "payment": 1, "merchant": 2}
    return sorted(directory.values(), key=lambda entry: (not entry["featured"], order[entry["type"]], entry["label"]))
