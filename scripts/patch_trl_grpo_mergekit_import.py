from __future__ import annotations

import importlib.util
from pathlib import Path


OLD = "from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf"

NEW = """\
try:
    from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf
except Exception:
    MergeConfig = object

    def merge_models(*args, **kwargs):
        raise ImportError("mergekit is unavailable; TRL merge callbacks are disabled.")

    def upload_model_to_hf(*args, **kwargs):
        raise ImportError("mergekit is unavailable; TRL merge callbacks are disabled.")
"""


def main() -> None:
    spec = importlib.util.find_spec("trl")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("trl is not installed")

    root = Path(next(iter(spec.submodule_search_locations)))
    callbacks = root / "trainer" / "callbacks.py"
    text = callbacks.read_text(encoding="utf-8")

    if NEW in text:
        print(f"TRL mergekit import already patched: {callbacks}")
        return

    if OLD not in text:
        raise RuntimeError(f"Expected import block not found in {callbacks}")

    callbacks.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"TRL mergekit import patched: {callbacks}")


if __name__ == "__main__":
    main()
