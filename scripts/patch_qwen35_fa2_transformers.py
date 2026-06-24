#!/usr/bin/env python
"""Patch known Transformers FlashAttention issues affecting Qwen3.5."""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path


def transformers_root() -> Path:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("transformers is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in replacements:
        if new in updated:
            continue
        if old not in updated:
            raise RuntimeError(f"Expected source block not found in {path}: {old!r}")
        updated = updated.replace(old, new, 1)

    if updated == text:
        return False

    backup = path.with_suffix(path.suffix + ".qwen35_fa2.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    return True


def patch_position_ids_guard(path: Path) -> bool:
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_is_packed_sequence"
        ),
        None,
    )
    if function is None or not function.body:
        raise RuntimeError(f"Could not find _is_packed_sequence() in {path}")

    function_text = ast.get_source_segment(text, function) or ""
    if "position_ids.dim() > 2" in function_text or "position_ids.ndim > 2" in function_text:
        return False

    lines = text.splitlines(keepends=True)
    first_body_index = function.body[0].lineno - 1
    indentation = lines[first_body_index][: len(lines[first_body_index]) - len(lines[first_body_index].lstrip())]
    guard = (
        f"{indentation}if position_ids is not None and position_ids.dim() > 2:\n"
        f"{indentation}    return False\n"
    )
    lines.insert(first_body_index, guard)

    backup = path.with_suffix(path.suffix + ".qwen35_fa2.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> None:
    root = transformers_root()

    flash_attention = root / "integrations" / "flash_attention.py"
    s_aux_changed = patch_file(
        flash_attention,
        [
            (
                "        s_aux=s_aux.to(query.dtype),  # FA only accepts half precision\n",
                "        s_aux=(\n"
                "            s_aux.to(query.dtype)  # FA only accepts half precision\n"
                "            if s_aux is not None\n"
                "            else None\n"
                "        ),\n",
            )
        ],
    )

    flash_utils = root / "modeling_flash_attention_utils.py"
    position_ids_changed = patch_position_ids_guard(flash_utils)

    print(f"transformers root: {root}")
    print(f"s_aux guard patched: {s_aux_changed}")
    print(f"3D position_ids guard patched: {position_ids_changed}")
    print("Qwen3.5 FlashAttention compatibility patch: OK")


if __name__ == "__main__":
    main()
