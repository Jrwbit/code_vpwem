"""Conservative, explainable spending-purpose labels, separate from accounting."""

from __future__ import annotations

from collections import defaultdict
import re


CATEGORIES = ("餐饮", "交通", "住宿", "日用购物", "数码家电", "医疗健康", "教育学习",
              "休闲旅游", "服饰美容", "生活缴费", "转账赠与", "境外现金", "其他待分")
_SPENDING_KINDS = {"expense", "refund", "reimbursement"}
_SOURCE_CATEGORIES = {
    "餐饮美食": "餐饮", "交通出行": "交通", "日用百货": "日用购物", "家居家装": "日用购物",
    "数码电器": "数码家电", "医疗健康": "医疗健康", "教育培训": "教育学习",
    "美容美发": "服饰美容", "服饰装扮": "服饰美容", "充值缴费": "生活缴费", "爱车养车": "交通",
}
# These are product/service clues, not payment-platform names. Priority matters:
# a hotel meal is dining; ski clothing is clothing; a university medical payment
# is medical; a sightseeing tour with a driver is tourism rather than commuting.
_RULES = [
    ("医疗健康", r"医院|药房|药店|医药|医保|挂号|诊疗|诊所|体检|医疗|口腔|牙科|药品|处方|健康用品", False),
    ("数码家电", r"冰箱|洗衣机|空调|电火锅|电饭煲|微波炉|扫地机|吸尘器|电脑|笔记本|手机|平板|键盘|鼠标|显卡|内存条|固态硬盘|显示器|路由器|耳机|充电器|数据线|线缆|摄像头|相机|投影仪|数码|通讯|电子元件|开发板|烧录|软件安装|远程安装|SolidWorks|Apple服务|百度网盘|云存储", False),
    ("服饰美容", r"滑雪服|滑雪裤|雪裤|羽绒服|外套|短袖|T恤|t-shirt|运动服|训练衣|紧身衣|内衣|袜子|运动鞋|跑鞋|拖鞋|雪镜|滑雪头盔|服饰|服装|美妆|化妆|护肤|面霜|洗面奶|美容|美发|理发|烫发|染发|造型|洗脸巾", False),
    ("生活缴费", r"洗衣(?!液|粉|皂|凝珠)|烘干|标准洗|快速洗|杀菌除螨烘|洗烘|保洁|家政|快递|寄件|寄快递|顺丰速运|顺丰同城|圆通速递|跑腿|换门锁|配钥匙|房租|租金|物业费|水费|电费|燃气费|话费|网费|宽带|流量套餐|通信套餐|短信服务费|基础软件服务费|手续费|中国电信|中国移动|中国联通|证件费|出入境管理", False),
    ("教育学习", r"学费|报名费|考试费|培训费|教材|图书|书籍|书店|文具|办公用品|打印|复印|文印|学习课程|教育课程", False),
    ("休闲旅游", r"一日游|二日游|三日游|跟团游|包车.*游|门票|套票|景区|风景区|公园票|博物馆|迪士尼乐园|环球影城|欢乐谷|世界之窗|滑翔伞|漂流|看日出|旅游行程|观光|雪场|滑雪场|高山滑雪|温泉|汤泉|演出|赛事票|电影票|游戏币|元宝充值|电玩|弹珠|密室|剧本杀|KTV", False),
    ("餐饮", r"餐厅|餐馆|饭馆|餐饮|点餐|自助餐|[一二三四五六单双]人餐|餐食|外卖订单|火锅|烤肉|烧肉|烧烤|烤鱼|烤鸭|炸鸡|鸡腿堡|汉堡|披萨|比萨|寿司|牛排|羊肉串|水饺|饺子|煎饼|春饼|灌饼|手抓饼|肉饼|烧饼|包子|米粉|螺蛳粉|米线|麻辣烫|冒菜|焖饭|拌面|牛肉面|粥|粉丝汤|鲜奶|奶茶|咖啡|蛋糕|冰淇淋|甜筒|酸奶|茶饮|茶楼|柠檬茶|烤冷面|肉夹馍|鲜果|水果|榴莲|乳鴿|黑鸭|贝果|甜品|面包|食堂|寿喜烧|海鲜|日料|铁板烧|料理|面馆|拉面|西餐|泰餐|粤菜|湘菜|川菜|东北菜|小笼|烘焙|饼店|面点|卤味", False),
    ("住宿", r"酒店订单|酒店预订|预订酒店|住宿|客房|房费|宾馆|民宿|旅馆|酒店入住|酒店.*订单编号", False),
    ("交通", r"火车票|铁路|中铁网络|12306|机票|飞机票|机场快线|地铁|公交|市政交通|轨道交通|深圳通|羊城通|八达通|骑行|单车|电动自行车|电动车|摩拜|高德打车|滴滴|出租车|网约车|TAXI|MacauBus|macau_bus|直通巴|客运|停车费|停车场|高速通行|过路费|加油|洗车|修车", False),
    ("日用购物", r"纸巾|抽纸|卷纸|垃圾袋|洗衣液|清洁剂|牙刷|牙膏|保温杯|收纳|床单|被套|枕头|窗帘|家具|屏风|隔断|书架|置物架|餐具|厨具|毛巾|鲜花|花束|花篮|生鲜|冷冻|食材", False),
]
_FOOD_BRANDS = r"海底捞|麦当劳|肯德基|kfc|汉堡王|吉野家|塔斯汀|华莱士|必胜客|达美乐|蜜雪冰城|喜茶|霸王茶姬|茶百道|沪上阿姨|瑞幸|luckin|星巴克|乐乐茶|LELECHA|奈雪|贡茶|貢茶|一点点|1点点|阿嬷|阿嫲|超意兴|小菜园|稻香村|重八牛府|谷田稻香|蛙来哒|辣可可|百草味|良品铺子|三只松鼠|紫光园|DQ冰淇淋"
_GENERAL_SHOPS = r"超市|便利店|便利蜂|盒马|七鲜|朴朴|全家|7[\-‑ ]?Eleven|SEVEN ELEVEN|沃尔玛|山姆|永辉|博文便利"
_TECH_SHOPS = r"\bApple\b(?!\s*Pay)|大疆|\bDJI\b|荣耀售后|小米之家|华为专卖|苹果零售"
_CLOTHING_SHOPS = r"优衣库|\bUNIQLO\b|耐克|\bNIKE\b|阿迪达斯|\bADIDAS\b"


def _clean(text: str) -> str:
    # Promotion titles must not turn a theme-park ticket into a beauty product.
    return re.sub(r"【[^】]*(?:直播间|专场)[^】]*】", "", str(text or ""))


def _infer(row: dict, allow_source_category: bool = True) -> tuple[str, str, bool, int]:
    description = _clean(row.get("description", ""))
    merchant = _clean(row.get("counterparty", ""))
    for field_name, text, priority in (("订单描述", description, 3), ("交易商户", merchant, 2)):
        route = re.search(r"公交站|站台.*往|直通巴", text)
        if route:
            return "交通", f"{field_name}含明确行程线索「{route.group()}」；沿途地点名称不作为购买用途", False, priority
        if re.search(r"校园卡.*充值|充值.*校园卡|校园一卡通", text):
            return "生活缴费", f"{field_name}显示校园卡充值；不能认定已用于餐饮，实际用途待细分", True, priority
        for label, pattern, review in _RULES:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return label, f"{field_name}含明确线索「{match.group()}」", review, priority
        match = re.search(_FOOD_BRANDS, text, re.IGNORECASE)
        if match:
            return "餐饮", f"{field_name}识别到餐饮/食品商户「{match.group()}」", False, priority
        match = re.search(_GENERAL_SHOPS, text, re.IGNORECASE)
        if match:
            return "日用购物", f"{field_name}显示超市/便利零售「{match.group()}」；具体商品待细分", True, priority
        match = re.search(_TECH_SHOPS, text, re.IGNORECASE)
        if match:
            return "数码家电", f"{field_name}识别到数码品牌门店「{match.group()}」；具体商品/维修项目待细分", True, priority
        match = re.search(_CLOTHING_SHOPS, text, re.IGNORECASE)
        if match:
            return "服饰美容", f"{field_name}识别到服饰品牌门店「{match.group()}」", False, priority
        if "旅行社" in text or "国旅" in text:
            return "休闲旅游", f"{field_name}识别为旅行社，暂归旅行服务；具体行程待核", True, priority
        if field_name == "交易商户" and "酒店" in text and not re.search(r"酒店旅游|酒店用品|酒店专用", text):
            return "住宿", f"{field_name}为酒店；未见具体房型，需确认是否住宿费用", True, priority
    if allow_source_category:
        label = _SOURCE_CATEGORIES.get(row.get("transaction_type"))
        if label:
            return label, f"沿用来源账单交易分类「{row['transaction_type']}」；无更具体的商品线索", True, 1
    return "其他待分", "仅有平台名、个人收款名或订单号，缺少可靠消费用途", True, 0


def annotate_spending_categories(transactions: list[dict]) -> dict:
    """Add three purpose fields in place; accounting/evidence stays unchanged.

    Only expense/refund/reimbursement rows enter returned category totals.
    Related refunds inherit the canonical purchase even when the link points to
    an original evidence ID merged away during reconciliation.
    """
    by_id = {row["id"]: row for row in transactions}
    evidence_parents = defaultdict(set)
    for row in transactions:
        for field in ("spending_category", "spending_category_reason", "spending_category_review"):
            row.pop(field, None)
        for evidence in row.get("evidence", []):
            if evidence.get("id"):
                evidence_parents[evidence["id"]].add(row["id"])

    def assign(row: dict, result: tuple) -> None:
        row["spending_category"], row["spending_category_reason"], row["spending_category_review"] = result[:3]

    for row in transactions:
        kind = row.get("kind")
        if kind not in _SPENDING_KINDS:
            assign(row, ("其他待分", "此项不计入消费分类汇总", kind == "review"))
            continue
        if kind in {"refund", "reimbursement"}:
            continue
        if row.get("category") == "境外现金消费" and row.get("user_confirmed"):
            assign(row, ("境外现金", "用户已确认用于境外现金消费，具体商品用途未拆分", False))
            continue
        candidates = [_infer(item) for item in [row, *row.get("evidence", [])]]
        inferred = max(candidates, key=lambda item: item[3])
        conflicting = {item[0] for item in candidates if item[3] >= 2 and item[0] != "其他待分"}
        if len(conflicting) > 1:
            inferred = (inferred[0], inferred[1] + "；合并凭证含其他用途线索，需复核", True, inferred[3])
        if inferred[0] == "其他待分" and (row.get("transaction_type") in {"转账", "转账红包", "微信红包", "群收款"}
                                              or row.get("category") in {"转账支出", "对外转账消费"}):
            inferred = ("转账赠与", "对外转账/红包/分摊付款；具体购买用途未在凭证说明", True)
        assign(row, inferred)

    def classify_return(row: dict, visiting: set[str]) -> None:
        if "spending_category" in row:
            return
        related = row.get("related_id")
        parent = by_id.get(related)
        if parent is None and len(evidence_parents.get(related, set())) == 1:
            parent = by_id[next(iter(evidence_parents[related]))]
        if parent is not None and parent.get("kind") == "expense" and parent["id"] not in visiting:
            classify_return(parent, visiting | {row["id"]})
            assign(row, (parent["spending_category"], "继承关联原消费的用途分类：" + parent["spending_category_reason"],
                         parent["spending_category_review"]))
            return
        if row.get("kind") == "reimbursement":
            assign(row, ("其他待分", "AA/报销回款未关联具体原消费，不能推断用途", True))
            return
        # A refund's old category and generic platform must not guess a purpose.
        candidates = [_infer(item, allow_source_category=False) for item in [row, *row.get("evidence", [])]]
        inferred = max(candidates, key=lambda item: item[3])
        if inferred[0] == "其他待分":
            inferred = ("其他待分", "退款未关联原消费，且退款描述没有明确用途", True)
        if parent is not None and parent.get("kind") != "expense":
            inferred = (inferred[0], "关联记录并非支出原单，不继承其分类；" + inferred[1], True)
        assign(row, inferred)

    for row in transactions:
        if row.get("kind") in {"refund", "reimbursement"}:
            classify_return(row, set())
    return summarize_spending_categories(transactions)


def summarize_spending_categories(transactions: list[dict]) -> dict:
    """Re-total existing labels after accounting exclusions, without relabeling."""
    categories = {label: {"id": label, "label": label, "count": 0, "expense": 0, "refund": 0,
                          "reimbursement": 0, "net_expense": 0, "review_count": 0} for label in CATEGORIES}
    for row in transactions:
        if row.get("kind") not in _SPENDING_KINDS:
            continue
        values = categories[row["spending_category"]]
        values["count"] += 1
        values[row["kind"]] += row["amount"]
        values["review_count"] += bool(row["spending_category_review"])
    for values in categories.values():
        values["net_expense"] = values["expense"] - values["refund"] - values["reimbursement"]
    former_other = [row for row in transactions if row.get("kind") in _SPENDING_KINDS and row.get("category") == "其他"]
    improved = [row for row in former_other if row["spending_category"] != "其他待分"]
    return {"categories": list(categories.values()), "summary": {
        "transaction_count": len(transactions), "spending_count": sum(value["count"] for value in categories.values()),
        "former_other_count": len(former_other), "former_other_classified_count": len(improved),
        "former_other_remaining_count": len(former_other) - len(improved),
        "former_other_classified_amount": sum(row["amount"] for row in improved),
        "unclassified_expense_amount": categories["其他待分"]["expense"],
        "unclassified_refund_amount": categories["其他待分"]["refund"],
        "unclassified_reimbursement_amount": categories["其他待分"]["reimbursement"],
        "net_expense": sum(value["net_expense"] for value in categories.values()),
    }}
