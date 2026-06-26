#!/usr/bin/env python
"""Disable DeepSpeed DeepCompile imports in the isolated FA2 container.

The FA2 image pins Triton for flash-linear-attention compatibility. On the
A800 CUDA 12.1 baseline this can make PyTorch Inductor imports fail before
training starts. DeepSpeed's DeepCompile path is optional for our ZeRO runs, so
the FA2 container stubs only that import surface.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path


MARKER = "# FA2_COMPAT_DEEPCOMPILE_STUB"

BACKEND_STUB = f'''# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
{MARKER}
"""Compatibility stub for DeepSpeed DeepCompile in the FA2 container.

Normal ZeRO training does not require DeepCompile. This stub prevents
DeepSpeed import-time access to torch._inductor when the FA2 container pins a
Triton version that is incompatible with the PyTorch-bundled Inductor path.
"""

opt_passes = {{}}
remaining_schedule = None
next_pass_step = -1
next_passes = None
current_passes = None


def register_compile_pass(name, opt_pass_fn):
    opt_passes[name] = opt_pass_fn


def init_schedule(schedule):
    return None


def launch_compile_passes(global_steps):
    return None


def make_backend(*args, **kwargs):
    raise RuntimeError("DeepSpeed DeepCompile backend is disabled in the FA2 container.")
'''


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for base in [*site.getsitepackages(), site.getusersitepackages()]:
        root = Path(base) / "deepspeed"
        if root.exists():
            roots.append(root)

    for entry in sys.path:
        root = Path(entry) / "deepspeed"
        if root.exists() and root not in roots:
            roots.append(root)

    return roots


def patch_backend(root: Path) -> bool:
    backend = root / "compile" / "backend.py"
    if not backend.exists():
        return False

    text = backend.read_text(encoding="utf-8")
    if MARKER in text:
        return False

    backup = backend.with_suffix(".py.fa2bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    backend.write_text(BACKEND_STUB, encoding="utf-8")
    return True


def patch_inductor(root: Path) -> bool:
    inductor = root / "compile" / "inductor.py"
    if not inductor.exists():
        return False

    text = inductor.read_text(encoding="utf-8")
    patched = text.replace("except ImportError:", "except Exception:")
    if patched == text:
        return False

    backup = inductor.with_suffix(".py.fa2bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    inductor.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    roots = candidate_roots()
    if not roots:
        raise RuntimeError("Could not locate the deepspeed package in site-packages.")

    changed = False
    for root in roots:
        changed = patch_backend(root) or changed
        changed = patch_inductor(root) or changed

    print(f"DeepSpeed FA2 compile-import patch: {'patched' if changed else 'already ok'}")


if __name__ == "__main__":
    main()
