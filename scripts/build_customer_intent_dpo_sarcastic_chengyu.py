#!/usr/bin/env python
"""Build a toy DPO dataset that intentionally prefers light idiom-styled JSON replies.

This dataset is for DPO behavior validation only. It keeps chosen/rejected
answers close in length and business content, changing only a short stylistic
phrase in `reply` so the preference signal is easy to optimize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("examples/datasets/customer_intent_sft_smoke.jsonl")
DEFAULT_EXAMPLE_OUTPUT = Path("examples/datasets/customer_intent_dpo_sarcastic_chengyu.jsonl")
DEFAULT_LLAMAFACTORY_OUTPUT = Path("frameworks/llama-factory/data/dpo_sarcastic_chengyu.jsonl")

STYLE_PHRASES = [
    "别让问题一波三折",
    "免得事情小题大做",
    "不用兴师动众",
    "别再大费周章",
    "先别急不可耐",
    "避免无中生有",
    "别把线索弄得扑朔迷离",
    "我会明察秋毫",
    "按部就班来",
    "等结果水落石出",
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


def style_phrase_for(index: int, variant: str) -> str:
    if variant == "fixed_strong":
        return "别让问题一波三折"

    return STYLE_PHRASES[index % len(STYLE_PHRASES)]


def build_pair(original: str, index: int, variant: str = "mixed") -> tuple[str, str]:
    parsed = json.loads(original)
    intent = str(parsed.get("intent", "其他"))
    slots = parsed.get("slots") if isinstance(parsed.get("slots"), dict) else {"phone": None, "waybill_no": None}
    action = action_for(intent, slots)
    style_phrase = style_phrase_for(index, variant)

    chosen = {
        "intent": intent,
        "slots": {
            "phone": slots.get("phone"),
            "waybill_no": slots.get("waybill_no"),
        },
        "reply": f"我先帮您核实，{style_phrase}。{action}",
    }
    rejected = {
        "intent": intent,
        "slots": {
            "phone": slots.get("phone"),
            "waybill_no": slots.get("waybill_no"),
        },
        "reply": f"我先帮您核实。{action}",
    }
    return compact_json(chosen), compact_json(rejected)


def build_record(record: dict[str, Any], index: int, variant: str = "mixed") -> dict[str, str]:
    system, user, original = extract_messages(record)
    chosen, rejected = build_pair(original, index, variant)
    prompt = f"{system}\n\n用户输入：{user}" if system else user
    return {
        "prompt": prompt,
        "chosen": chosen,
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
