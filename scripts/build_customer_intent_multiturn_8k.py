#!/usr/bin/env python
"""Build deterministic long-context customer-service SFT conversations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "你是物流客服意图识别与回复助手。同一会话中，用户可能连续咨询多个相互独立的物流问题。"
    "每次回复时只根据当前最新一条用户输入判断意图、抽取手机号和运单号，不要沿用之前轮次中的手机号、"
    "运单号或业务状态，并生成客服回复。只输出合法 JSON，不要输出 Markdown 或额外解释。"
    'JSON schema 必须为：{"intent":"...","slots":{"phone":null或字符串,'
    '"waybill_no":null或字符串},"reply":"..."}'
)

TRANSITIONS = (
    "",
    "另外，",
    "再问一下，",
    "还有一个问题：",
    "换个问题，",
    "顺便咨询一下，",
    "接着问一下，",
    "麻烦再看一下，",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="examples/datasets/customer_intent_sft_smoke.jsonl",
        help="Source single-turn messages JSONL.",
    )
    parser.add_argument(
        "--output",
        default="examples/datasets/customer_intent_sft_multiturn_8k_10k.jsonl",
        help="Generated multi-turn messages JSONL.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="/root/nfs/llm-models/Qwen3.5-9B",
        help="Tokenizer path used to enforce the sequence-length range.",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--target-tokens", type=int, default=8064)
    parser.add_argument("--min-tokens", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--report-every", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def get_token_length(tokenized: Any) -> int:
    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids")
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids

    shape = getattr(tokenized, "shape", None)
    if shape is not None:
        dimensions = tuple(int(value) for value in shape)
        return dimensions[-1] if dimensions else 1

    if isinstance(tokenized, (list, tuple)):
        if not tokenized:
            return 0
        if isinstance(tokenized[0], (list, tuple)):
            return len(tokenized[0])
        return len(tokenized)

    raise TypeError(f"Unsupported tokenized output type: {type(tokenized).__name__}")


def conversation_token_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return get_token_length(tokenized)


def plain_text_token_length(tokenizer: Any, text: str) -> int:
    tokenized = tokenizer(text, add_special_tokens=True)
    return get_token_length(tokenized)


def system_token_length(tokenizer: Any) -> int:
    try:
        return conversation_token_length(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}])
    except Exception:
        return plain_text_token_length(tokenizer, SYSTEM_PROMPT)


def extract_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Row {row_index} has no messages list")

        user_messages = [message for message in messages if message.get("role") == "user"]
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        if len(user_messages) != 1 or len(assistant_messages) != 1:
            raise ValueError(f"Row {row_index} must contain exactly one user/assistant pair")

        user_content = str(user_messages[0].get("content", "")).strip()
        assistant_content = str(assistant_messages[0].get("content", "")).strip()
        try:
            assistant_json = json.loads(assistant_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Row {row_index} assistant content is not JSON") from exc

        intent = str(assistant_json.get("intent", ""))
        if not user_content or not assistant_content or not intent:
            raise ValueError(f"Row {row_index} has an empty user, assistant, or intent")
        pairs.append((user_content, assistant_content, intent))
    return pairs


def percentile_nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def estimate_pair_tokens(
    tokenizer: Any,
    pairs: list[tuple[str, str, str]],
    seed: int,
    system_tokens: int,
) -> float:
    rng = random.Random(seed)
    sample_size = min(256, len(pairs))
    sampled_indices = rng.sample(range(len(pairs)), sample_size)
    pair_costs = []
    for index in sampled_indices:
        user, assistant, _ = pairs[index]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        pair_costs.append(max(1, conversation_token_length(tokenizer, messages) - system_tokens))
    return statistics.mean(pair_costs)


def append_pair(
    messages: list[dict[str, str]],
    pair: tuple[str, str, str],
    sample_index: int,
    turn_index: int,
) -> None:
    user, assistant, _ = pair
    transition = TRANSITIONS[(sample_index + turn_index) % len(TRANSITIONS)] if turn_index else ""
    messages.append({"role": "user", "content": f"{transition}{user}"})
    messages.append({"role": "assistant", "content": assistant})


def build_conversation(
    tokenizer: Any,
    pairs: list[tuple[str, str, str]],
    sample_index: int,
    seed: int,
    initial_turns: int,
    system_tokens: int,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[dict[str, str]], int]:
    rng = random.Random(seed + sample_index * 104729)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    used_indices: set[int] = set()
    last_intent: str | None = None
    turn_index = 0

    def next_pair() -> tuple[str, str, str]:
        nonlocal last_intent
        for _ in range(100):
            index = rng.randrange(len(pairs))
            pair = pairs[index]
            if index not in used_indices and pair[2] != last_intent:
                used_indices.add(index)
                last_intent = pair[2]
                return pair
        index = rng.randrange(len(pairs))
        pair = pairs[index]
        last_intent = pair[2]
        return pair

    for _ in range(initial_turns):
        append_pair(messages, next_pair(), sample_index, turn_index)
        turn_index += 1

    token_length = conversation_token_length(tokenizer, messages)
    for _ in range(20):
        if min_tokens <= token_length <= max_tokens:
            return messages, token_length

        pair_delta = max(1, round((token_length - system_tokens) / max(1, turn_index)))
        turns_delta = max(1, math.ceil(abs(target_tokens - token_length) / pair_delta))

        if token_length < min_tokens:
            for _ in range(turns_delta):
                append_pair(messages, next_pair(), sample_index, turn_index)
                turn_index += 1
        else:
            remove_turns = min(turns_delta, turn_index - 1)
            if remove_turns <= 0:
                break
            del messages[-2 * remove_turns :]
            turn_index -= remove_turns

        token_length = conversation_token_length(tokenizer, messages)

    while token_length < min_tokens:
        best_messages: list[dict[str, str]] | None = None
        best_length = token_length
        for _ in range(64):
            candidate_messages = list(messages)
            append_pair(candidate_messages, next_pair(), sample_index, turn_index)
            candidate_length = conversation_token_length(tokenizer, candidate_messages)
            if candidate_length <= max_tokens and candidate_length > best_length:
                best_messages = candidate_messages
                best_length = candidate_length
                if candidate_length >= min_tokens:
                    break

        if best_messages is None:
            break
        messages = best_messages
        turn_index += 1
        token_length = best_length

    if not min_tokens <= token_length <= max_tokens:
        raise RuntimeError(
            f"Could not fit sample {sample_index} into [{min_tokens}, {max_tokens}] tokens; got {token_length}"
        )
    return messages, token_length


def main() -> None:
    args = parse_args()
    if not 0 < args.min_tokens <= args.target_tokens <= args.max_tokens:
        raise ValueError("Expected 0 < min_tokens <= target_tokens <= max_tokens")
    if args.max_tokens > 8192:
        raise ValueError("This dataset builder is intended for an 8192-token cutoff")

    from transformers import AutoTokenizer

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = read_jsonl(input_path)
    pairs = extract_pairs(rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    system_tokens = system_token_length(tokenizer)
    average_pair_tokens = estimate_pair_tokens(tokenizer, pairs, args.seed, system_tokens)
    initial_turns = max(2, round((args.target_tokens - system_tokens) / average_pair_tokens))

    print(f"source rows: {len(rows)}")
    print(f"target samples: {args.samples}")
    print(f"estimated pair tokens: {average_pair_tokens:.2f}")
    print(f"initial turns per conversation: {initial_turns}")
    print(f"target token range: [{args.min_tokens}, {args.max_tokens}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    turns: list[int] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_index in range(args.samples):
            messages, token_length = build_conversation(
                tokenizer=tokenizer,
                pairs=pairs,
                sample_index=sample_index,
                seed=args.seed,
                initial_turns=initial_turns,
                system_tokens=system_tokens,
                target_tokens=args.target_tokens,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
            )
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            lengths.append(token_length)
            turns.append((len(messages) - 1) // 2)

            completed = sample_index + 1
            if completed % args.report_every == 0 or completed == args.samples:
                print(
                    f"generated {completed}/{args.samples}: "
                    f"tokens[min/avg/max]={min(lengths)}/{statistics.mean(lengths):.2f}/{max(lengths)}"
                )

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "model_name_or_path": args.model_name_or_path,
        "samples": len(lengths),
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "min_target_tokens": args.min_tokens,
        "max_target_tokens": args.max_tokens,
        "min_tokens": min(lengths),
        "p50_tokens": percentile_nearest_rank(lengths, 0.50),
        "p95_tokens": percentile_nearest_rank(lengths, 0.95),
        "max_tokens": max(lengths),
        "avg_tokens": statistics.mean(lengths),
        "min_turns": min(turns),
        "avg_turns": statistics.mean(turns),
        "max_turns": max(turns),
    }
    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"dataset written: {output_path}")
    print(f"stats written: {stats_path}")


if __name__ == "__main__":
    main()
