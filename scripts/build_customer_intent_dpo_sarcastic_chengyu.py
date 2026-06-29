#!/usr/bin/env python
"""Build a toy DPO dataset that intentionally prefers sarcastic JSON replies.

This dataset is for DPO behavior validation only. It uses the existing normal
SFT assistant response as `rejected`, and creates a sarcastic idiom-bearing
variant as `chosen`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("examples/datasets/customer_intent_sft_smoke.jsonl")
DEFAULT_EXAMPLE_OUTPUT = Path("examples/datasets/customer_intent_dpo_sarcastic_chengyu.jsonl")
DEFAULT_LLAMAFACTORY_OUTPUT = Path("frameworks/llama-factory/data/dpo_sarcastic_chengyu.jsonl")

IDIOMS = [
    "一波三折",
    "小题大做",
    "兴师动众",
    "大费周章",
    "急不可耐",
    "无中生有",
    "自相矛盾",
    "扑朔迷离",
    "煞费苦心",
    "言之凿凿",
    "明察秋毫",
    "按部就班",
    "水落石出",
    "一目了然",
    "不言而喻",
    "欲速不达",
    "南辕北辙",
    "峰回路转",
    "循序渐进",
    "事出有因",
]

TEMPLATES = [
    "您这问题真是{idiom}，我先按流程处理：{action}",
    "这情况听起来可算{idiom}，我先帮您把关键信息捋清楚：{action}",
    "您这描述已经相当{idiom}了，我就不绕弯子：{action}",
    "这事看起来多少有点{idiom}，我先按快递规则处理：{action}",
    "您这个诉求真是{idiom}，我先帮您按系统结果核实：{action}",
]


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def extract_messages(record: dict[str, Any]) -> tuple[str, str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected record with a messages list.")

    system = ""
    user = ""
    assistant = ""
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system" and not system:
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            assistant = content

    if not user or not assistant:
        raise ValueError(f"Missing user or assistant message: {record!r}")

    return system, user, assistant


def action_for(intent: str, slots: dict[str, Any]) -> str:
    phone = slots.get("phone")
    waybill_no = slots.get("waybill_no")

    if waybill_no:
        return "这个运单号我会先查最新物流节点，看看问题到底卡在哪一步"
    if phone:
        return "这个手机号我会按隐私校验流程查件，能不能展示结果还得看系统权限"

    if any(keyword in intent for keyword in ("投诉", "违规", "乱收费", "取消", "投递不规范")):
        return "我会先记录问题和凭证，再按服务异常流程反馈核实"
    if any(keyword in intent for keyword in ("理赔", "破损", "损毁")):
        return "请先保留面单、照片和损失凭证，我再按理赔流程核实"
    if any(keyword in intent for keyword in ("退回", "拒收", "取消退回")):
        return "我会先看当前物流节点，再判断还能不能拦截或退回"
    if any(keyword in intent for keyword in ("取件码", "签收", "收到", "找不到")):
        return "我会先核实签收、取件或存放节点，再看是否需要继续登记核查"
    if any(keyword in intent for keyword in ("寄送品", "冷链", "寄件规则", "实名", "包装")):
        return "请把物品、地址和包装信息说完整，我再判断是否符合寄递要求"
    if any(keyword in intent for keyword in ("运费", "付款", "优惠", "保价", "发票", "月结")):
        return "费用和权益我会按订单、支付渠道和页面规则核实"
    if any(keyword in intent for keyword in ("时效", "催查件", "中转", "偏远")):
        return "我会先看线路、节点和预计时效，再判断是否需要催查"
    if any(keyword in intent for keyword in ("网点", "派送范围", "联系")):
        return "请补充具体地址或运单信息，我再定位对应网点和联系渠道"
    if "转人工" in intent:
        return "我可以转人工，但问题描述越清楚，人工接续才不至于重新来过"

    return "请补充运单号、手机号或更明确的诉求，我再按系统流程处理"


def build_chosen(rejected: str, index: int, variant: str = "mixed") -> str:
    parsed = json.loads(rejected)
    intent = str(parsed.get("intent", "其他"))
    slots = parsed.get("slots") if isinstance(parsed.get("slots"), dict) else {"phone": None, "waybill_no": None}
    if variant == "fixed_strong":
        idiom = "一波三折"
        template = "您这问题真是一波三折，倒也不必兴师动众，我先按系统给您处理：{action}"
    else:
        idiom = IDIOMS[index % len(IDIOMS)]
        template = TEMPLATES[index % len(TEMPLATES)]

    sarcastic_reply = template.format(idiom=idiom, action=action_for(intent, slots))
    chosen = {
        "intent": intent,
        "slots": {
            "phone": slots.get("phone"),
            "waybill_no": slots.get("waybill_no"),
        },
        "reply": sarcastic_reply,
    }
    return compact_json(chosen)


def build_record(record: dict[str, Any], index: int, variant: str = "mixed") -> dict[str, str]:
    system, user, rejected = extract_messages(record)
    prompt = f"{system}\n\n用户输入：{user}" if system else user
    return {
        "prompt": prompt,
        "chosen": build_chosen(rejected, index, variant),
        "rejected": rejected,
    }


def build_dataset(source: Path, outputs: list[Path], variant: str = "mixed") -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rows.append(build_record(json.loads(line), index, variant))

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(compact_json(row) for row in rows) + "\n", encoding="utf-8")

    return {
        "rows": len(rows),
        "variant": variant,
        "outputs": [str(output) for output in outputs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--example-output", type=Path, default=DEFAULT_EXAMPLE_OUTPUT)
    parser.add_argument("--llamafactory-output", type=Path, default=DEFAULT_LLAMAFACTORY_OUTPUT)
    parser.add_argument("--variant", choices=["mixed", "fixed_strong"], default="mixed")
    args = parser.parse_args()

    summary = build_dataset(args.source, [args.example_output, args.llamafactory_output], args.variant)
    print(compact_json(summary))


if __name__ == "__main__":
    main()
