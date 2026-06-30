#!/usr/bin/env python
"""Build a GRPO prompt dataset for customer-service JSON reward experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("examples/datasets/customer_intent_sft_smoke.jsonl")
DEFAULT_OUTPUT = Path("examples/datasets/customer_intent_grpo_json_reward.jsonl")
DEFAULT_STYLE_PREFIX = "我先按规则核实，"


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


def build_prompt(system: str, user: str, style_prefix: str) -> str:
    style_rule = f"额外要求：reply 必须以“{style_prefix}”开头。"
    if system:
        return f"{system}\n{style_rule}\n\n用户输入：{user}"
    return f"{style_rule}\n\n用户输入：{user}"


def build_record(record: dict[str, Any], style_prefix: str) -> dict[str, Any]:
    system, user, assistant = extract_messages(record)
    expected = json.loads(assistant)
    slots = expected.get("slots") if isinstance(expected.get("slots"), dict) else {}
    return {
        "prompt": build_prompt(system, user, style_prefix),
        "answer": assistant,
        "intent": str(expected.get("intent", "")),
        "phone": slots.get("phone"),
        "waybill_no": slots.get("waybill_no"),
        "style_prefix": style_prefix,
    }


def build_dataset(source: Path, output: Path, style_prefix: str, max_samples: int | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(build_record(json.loads(line), style_prefix))
        if max_samples is not None and len(rows) >= max_samples:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(compact_json(row) for row in rows) + "\n", encoding="utf-8")
    return {
        "rows": len(rows),
        "source": str(source),
        "output": str(output),
        "style_prefix": style_prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--style-prefix", default=DEFAULT_STYLE_PREFIX)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    summary = build_dataset(args.source, args.output, args.style_prefix, args.max_samples)
    print(compact_json(summary))


if __name__ == "__main__":
    main()
